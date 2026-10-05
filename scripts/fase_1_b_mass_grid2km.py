#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motor de Extracción Masiva Nativa 2 km GOES-16 (Convenio ATLAS II)
=================================================================
Descarga en streaming y extrae la serie satelital histórica directamente sobre
las 59,837 celdas de la Provincia de Córdoba con proyección canónica PROJ / pyproj.

Características de Producción y Portabilidad:
1. Cero Dependencia Local:
   - Si existen índices locales (data/goes/YYYYMMDD/_index.csv), los aprovecha como caché ultra rápida.
   - Si NO existen (p. ej. en otra computadora o instalación limpia), descubre y lista las escenas
     directamente sobre NOAA S3 al vuelo (boto3 list_objects_v2, sin credenciales AWS).
2. Paralelismo Multinodo / Multimáquina:
   - Soporta particionamiento por rangos de fecha (--start / --end).
   - Soporta particionamiento por años (--years 2020,2021,2022 vs 2023,2024,2025).
   - Soporta sharding intercalado automático (--shard 0 --total-shards 2).
3. Configuración de Transferencia S3 Óptima:
   - 4 workers x 4 threads multipart vía TransferConfig (22.3 MB/s agregados).
4. Verificación Estricta de Integridad:
   - Comprobación exacta de ContentLength contra S3 por cada descarga.
5. Escritura Incremental con ParquetWriter (RAM O(1)):
   - Escribe cada escena como un row group independiente en el archivo temporal diario.
   - La memoria RAM se mantiene estrictamente constante.
6. Bounding Box Dinámico ABI:
   - Buffer de 2 píxeles reales ABI (~28 microradianes) alrededor de Córdoba.
7. Detección Física Adaptativa Diurna:
   - Selección de escenas según cenit solar en Córdoba (zenith < 87.0°),
     adaptándose dinámicamente entre verano (~27 escenas) e invierno (~18 escenas).
8. Integridad Bloqueante de Días:
   - Si alguna escena esperada falla, el día se marca como PARTIAL, se borra el
     archivo temporal y se reporta en metadata/failed_scenes_2km.jsonl.
9. Compresión ZSTD Level 3:
   - ~21-35 MB por día, ocupando sólo ~46 GB para los 6 años completos.
"""

import os
import sys
import re
import time
import json
import argparse
import signal
import shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
import pvlib
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from boto3.s3.transfer import TransferConfig

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GRID_FILE = ROOT_DIR / "data" / "ready" / "cordoba_grid_2km_regions.parquet"
DEFAULT_OUT_DIR = ROOT_DIR / "data" / "processed_2km"
DEFAULT_TMP_DIR = ROOT_DIR / "data" / ".tmp_download"
DEFAULT_INDEX_DIR = ROOT_DIR / "data" / "goes"
DEFAULT_FAILED_LOG = ROOT_DIR / "metadata" / "failed_scenes_2km.jsonl"

ESUN_C02 = 663.274  # W/(m^2 um) para Banda 2 (0.64 um)
CBA_CENTROID_LAT = -32.5
CBA_CENTROID_LON = -63.5

# Esquema oficial de PyArrow para escritura en streaming
SCHEMA_2KM = pa.schema([
    ("cell_id", pa.int32()),
    ("lon", pa.float32()),
    ("lat", pa.float32()),
    ("radiance", pa.float32()),
    ("zenith_deg", pa.float32()),
    ("azimuth_deg", pa.float32()),
    ("reflectance", pa.float32()),
    ("timestamp", pa.timestamp("us", tz="UTC"))
])

# Variables globales inicializadas una sola vez por proceso worker
_WORKER_GRID_DF = None
_WORKER_X_SCAN = None
_WORKER_Y_SCAN = None
_WORKER_S3_CLIENT = None
_WORKER_TRANSFER_CONFIG = None


def init_worker(grid_file: Path, concurrency_per_file: int = 4):
    """Inicializador de worker: carga grilla, precomputa scan coords y crea cliente S3 tunado."""
    global _WORKER_GRID_DF, _WORKER_X_SCAN, _WORKER_Y_SCAN, _WORKER_S3_CLIENT, _WORKER_TRANSFER_CONFIG

    if not Path(grid_file).exists():
        raise FileNotFoundError(f"Archivo de grilla 2 km no encontrado: {grid_file}")

    _WORKER_GRID_DF = pd.read_parquet(grid_file)

    h = 35786023.0
    a = 6378137.0
    b = 6356752.31414
    lon_0 = -75.0
    sweep = "x"

    p_geos = pyproj.Proj(proj="geos", h=h, a=a, b=b, lon_0=lon_0, sweep=sweep)
    lons = _WORKER_GRID_DF["lon"].values
    lats = _WORKER_GRID_DF["lat"].values

    X, Y = p_geos(lons, lats)
    _WORKER_X_SCAN = (X / h).astype(np.float64)
    _WORKER_Y_SCAN = (Y / h).astype(np.float64)

    _WORKER_S3_CLIENT = boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            retries={"max_attempts": 5, "mode": "adaptive"},
            max_pool_connections=32,
            tcp_keepalive=True,
        )
    )

    _WORKER_TRANSFER_CONFIG = TransferConfig(
        multipart_threshold=8 * 1024**2,
        multipart_chunksize=16 * 1024**2,
        max_concurrency=concurrency_per_file,
        use_threads=True,
    )


def parse_s3_url(url: str):
    p = urlparse(url)
    bucket = p.netloc
    key = p.path.lstrip("/")
    if key.startswith(bucket + "/"):
        key = key[len(bucket) + 1 :]
    return bucket, key


def download_with_retry_and_verify(bucket: str, key: str, local_dest: Path, max_retries: int = 3) -> bool:
    """Descarga de S3 con TransferConfig y verificación exacta de ContentLength."""
    global _WORKER_S3_CLIENT, _WORKER_TRANSFER_CONFIG

    for attempt in range(max_retries):
        try:
            head = _WORKER_S3_CLIENT.head_object(Bucket=bucket, Key=key)
            expected_size = head.get("ContentLength")

            _WORKER_S3_CLIENT.download_file(
                bucket,
                key,
                str(local_dest),
                Config=_WORKER_TRANSFER_CONFIG
            )

            if local_dest.exists():
                actual_size = local_dest.stat().st_size
                if expected_size is None or actual_size == expected_size:
                    return True
        except Exception:
            if attempt == max_retries - 1:
                return False
            time.sleep(2 ** attempt)
    return False


def extract_scene_from_local_netcdf(
    nc_path: Path,
    scene_time: pd.Timestamp
) -> pd.DataFrame:
    """Extrae el recorte de Córdoba y calcula radiancia, reflectancia y geometría solar."""
    global _WORKER_GRID_DF, _WORKER_X_SCAN, _WORKER_Y_SCAN

    with xr.open_dataset(nc_path, engine="h5netcdf") as ds:
        # Buffer adaptativo: 2 píxeles ABI reales
        dx = abs(float(ds.x[1] - ds.x[0]))
        dy = abs(float(ds.y[1] - ds.y[0]))
        buffer_x = 2.0 * dx
        buffer_y = 2.0 * dy

        x_min, x_max = float(_WORKER_X_SCAN.min() - buffer_x), float(_WORKER_X_SCAN.max() + buffer_x)
        y_min, y_max = float(_WORKER_Y_SCAN.min() - buffer_y), float(_WORKER_Y_SCAN.max() + buffer_y)

        y_slice = slice(y_max, y_min) if ds.y[0] > ds.y[-1] else slice(y_min, y_max)
        x_slice = slice(x_min, x_max) if ds.x[0] < ds.x[-1] else slice(x_max, x_min)

        rad_crop = ds["Rad"].sel(x=x_slice, y=y_slice).load()

        rad_pts = rad_crop.interp(
            x=("points", _WORKER_X_SCAN),
            y=("points", _WORKER_Y_SCAN),
            method="linear"
        ).values.astype(np.float32)

    lats = _WORKER_GRID_DF["lat"].values
    lons = _WORKER_GRID_DF["lon"].values
    n_pts = len(lats)
    t_series = pd.Series(scene_time, index=range(n_pts))

    sp = pvlib.solarposition.get_solarposition(t_series, lats, lons)
    zenith_deg = sp["zenith"].values.astype(np.float32)
    azimuth_deg = sp["azimuth"].values.astype(np.float32)

    cosz = np.cos(np.deg2rad(zenith_deg))
    reflectance = np.where(cosz > 0.0, (np.pi * rad_pts) / (ESUN_C02 * cosz), np.nan)
    reflectance = np.clip(reflectance, 0.0, 1.2).astype(np.float32)

    return pd.DataFrame({
        "cell_id": _WORKER_GRID_DF["cell_id"].values.astype(np.int32),
        "lon": lons.astype(np.float32),
        "lat": lats.astype(np.float32),
        "radiance": rad_pts,
        "zenith_deg": zenith_deg,
        "azimuth_deg": azimuth_deg,
        "reflectance": reflectance,
        "timestamp": scene_time
    })


def select_daylight_scenes_from_df(df_raw: pd.DataFrame, minute_step: int = 30) -> pd.DataFrame:
    """Aplica el filtrado físico adaptativo por cenit solar y ranurado temporal (10, 30, 60 min)."""
    if df_raw.empty:
        return pd.DataFrame()

    df_idx = df_raw.copy()
    df_idx["ts"] = pd.to_datetime(df_idx["ts"], utc=True)

    df_idx = df_idx.sort_values("ts").reset_index(drop=True)

    sp = pvlib.solarposition.get_solarposition(df_idx["ts"], CBA_CENTROID_LAT, CBA_CENTROID_LON)
    df_idx["zenith_centroid"] = sp["zenith"].values
    df_idx = df_idx[df_idx["zenith_centroid"] < 87.0].copy()

    if df_idx.empty:
        return pd.DataFrame()

    if minute_step == 30:
        df_idx["slot"] = df_idx["ts"].dt.round("30min")
        df_idx["diff_sec"] = np.abs((df_idx["ts"] - df_idx["slot"]).dt.total_seconds())
        df_idx = df_idx[df_idx["diff_sec"] <= 420]
        df_idx = df_idx.sort_values("diff_sec").drop_duplicates(subset=["slot"]).sort_values("ts")
    elif minute_step == 60:
        df_idx["slot"] = df_idx["ts"].dt.round("1h")
        df_idx["diff_sec"] = np.abs((df_idx["ts"] - df_idx["slot"]).dt.total_seconds())
        df_idx = df_idx[df_idx["diff_sec"] <= 420]
        df_idx = df_idx.sort_values("diff_sec").drop_duplicates(subset=["slot"]).sort_values("ts")
    elif minute_step == 10:
        pass
    else:
        raise ValueError(f"minute_step {minute_step} no soportado. Use 10, 30 o 60.")

    return df_idx.reset_index(drop=True)


def get_daylight_scenes_for_day(
    day_str: str,
    minute_step: int = 30,
    index_dir: Path = None,
    s3_client = None
) -> pd.DataFrame:
    """
    Resuelve las escenas diurnas para el día indicado de forma robusta y portátil:
    1. Si existe un archivo local index_dir/day_str/_index.csv, lo lee directamente (máxima velocidad).
    2. Si NO existe localmente (otra máquina, clon limpio, etc.), consulta NOAA S3 directamente
       al vuelo vía list_objects_v2, extrayendo las escenas diurnas de Córdoba en 2-3 segundos.
    """
    # 1. Intentar índice local si está disponible
    if index_dir is not None:
        local_idx_file = Path(index_dir) / day_str / "_index.csv"
        if local_idx_file.exists():
            try:
                df_idx = pd.read_csv(local_idx_file)
                if "band" in df_idx.columns:
                    df_idx = df_idx[df_idx["band"] == "C02"]
                if "sat" in df_idx.columns:
                    df_idx = df_idx[df_idx["sat"].isin(["goes16", "goes19"])]
                df_idx = df_idx[df_idx["url"].str.contains(r"noaa-goes(?:16|19)", regex=True, na=False)]
                if not df_idx.empty:
                    df_idx["ts"] = pd.to_datetime(df_idx["start_iso"], utc=True)
                    df_res = select_daylight_scenes_from_df(df_idx, minute_step=minute_step)
                    if not df_res.empty:
                        return df_res
            except Exception:
                pass

    # 2. Descubrimiento directo al vuelo en NOAA S3 (autónomo y multi-satélite G16/G19)
    if s3_client is None:
        s3_client = boto3.client(
            "s3",
            config=Config(
                signature_version=UNSIGNED,
                retries={"max_attempts": 5, "mode": "adaptive"}
            )
        )

    dt = datetime.strptime(day_str, "%Y%m%d")
    doy = dt.timetuple().tm_yday

    # Identificar horas diurnas para Córdoba (zenith < 88°)
    hours_dt = pd.date_range(
        f"{dt.strftime('%Y-%m-%d')} 00:00",
        f"{dt.strftime('%Y-%m-%d')} 23:00",
        freq="1h",
        tz="UTC"
    )
    sp = pvlib.solarposition.get_solarposition(hours_dt, CBA_CENTROID_LAT, CBA_CENTROID_LON)
    daylight_hours = hours_dt[sp["zenith"] < 88.0].hour.tolist()

    if not daylight_hours:
        return pd.DataFrame()

    # Selección dinámica de satélite/bucket según fecha de traspaso operativo (8 de abril de 2025 = DOY 98)
    if dt.year < 2025 or (dt.year == 2025 and doy < 92):
        sat_candidates = [("noaa-goes16", "C02_G16", "goes16")]
    elif dt.year == 2025 and doy <= 97:
        sat_candidates = [("noaa-goes16", "C02_G16", "goes16"), ("noaa-goes19", "C02_G19", "goes19")]
    else:
        # A partir del 8 de abril de 2025: GOES-19 operativo como GOES-East
        sat_candidates = [("noaa-goes19", "C02_G19", "goes19"), ("noaa-goes16", "C02_G16", "goes16")]

    records = []
    selected_sat = "goes16"

    for bucket, band_tag, sat_name in sat_candidates:
        cand_records = []
        for h in daylight_hours:
            prefix = f"ABI-L1b-RadF/{dt.year}/{doy:03d}/{h:02d}/"
            try:
                resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
                for obj in resp.get("Contents", []):
                    key = obj["Key"]
                    if band_tag in key:
                        filename = key.split("/")[-1]
                        m = re.search(r"_s(\d{13})", filename)
                        if m:
                            ts = datetime.strptime(m.group(1), "%Y%j%H%M%S").replace(tzinfo=timezone.utc)
                            cand_records.append({
                                "url": f"s3://{bucket}/{key}",
                                "ts": ts,
                                "sat": sat_name
                            })
            except Exception:
                continue

        if cand_records:
            records = cand_records
            selected_sat = sat_name
            break

    if not records:
        return pd.DataFrame()

    df_s3 = pd.DataFrame(records)
    df_selected = select_daylight_scenes_from_df(df_s3, minute_step=minute_step)

    # Opcional: escribir caché local si index_dir es válido
    if index_dir is not None and not df_selected.empty:
        try:
            cache_folder = Path(index_dir) / day_str
            cache_folder.mkdir(parents=True, exist_ok=True)
            cache_file = cache_folder / "_index.csv"
            if not cache_file.exists():
                cache_df = pd.DataFrame({
                    "url": df_selected["url"],
                    "start_iso": df_selected["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                    "band": "C02",
                    "sat": selected_sat
                })
                cache_df.to_csv(cache_file, index=False)
        except Exception:
            pass

    return df_selected


def process_single_day_worker(
    day_str: str,
    out_dir: Path,
    tmp_base_dir: Path,
    minute_step: int = 30,
    index_dir: Path = None
) -> dict:
    """Worker task con escritura incremental vía ParquetWriter (RAM O(1))."""
    global _WORKER_GRID_DF, _WORKER_S3_CLIENT

    out_file = out_dir / f"reflectancia_{day_str}.parquet"
    pid = os.getpid()
    tmp_out = out_dir / f"reflectancia_{day_str}.parquet.tmp_{pid}"

    if out_file.exists() and out_file.stat().st_size > 1024 * 1024:
        return {"status": "SKIPPED", "day": day_str, "scenes": 0, "time": 0.0, "failed_scenes": []}

    df_selected = get_daylight_scenes_for_day(
        day_str,
        minute_step=minute_step,
        index_dir=index_dir,
        s3_client=_WORKER_S3_CLIENT
    )

    if df_selected.empty:
        return {"status": "NO_SCENES_FOUND", "day": day_str, "scenes": 0, "time": 0.0, "failed_scenes": []}

    expected_scenes = len(df_selected)
    worker_tmp_dir = tmp_base_dir / f"worker_{pid}"
    worker_tmp_dir.mkdir(parents=True, exist_ok=True)

    failed_scenes = []
    downloaded_scenes = 0
    t0 = time.time()

    # Iniciar ParquetWriter incremental
    writer = pq.ParquetWriter(str(tmp_out), SCHEMA_2KM, compression="zstd", compression_level=3)

    try:
        for _, row in df_selected.iterrows():
            url = row["url"]
            scene_ts = row["ts"]
            bucket, key = parse_s3_url(url)
            temp_nc = worker_tmp_dir / Path(key).name

            try:
                ok = download_with_retry_and_verify(bucket, key, temp_nc)
                if not ok:
                    failed_scenes.append({"url": url, "timestamp": str(scene_ts), "error": "Download timeout/corrupt"})
                    continue

                df_s = extract_scene_from_local_netcdf(temp_nc, scene_ts)
                table = pa.Table.from_pandas(df_s, schema=SCHEMA_2KM, preserve_index=False)
                writer.write_table(table)
                downloaded_scenes += 1
                del df_s, table

            except Exception as e:
                failed_scenes.append({"url": url, "timestamp": str(scene_ts), "error": str(e)})
            finally:
                if temp_nc.exists():
                    temp_nc.unlink(missing_ok=True)
    finally:
        writer.close()

    # REGLA BLOQUEANTE: Si falta alguna escena esperada, descartar y eliminar
    if downloaded_scenes != expected_scenes:
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        return {
            "status": "PARTIAL",
            "day": day_str,
            "expected": expected_scenes,
            "downloaded": downloaded_scenes,
            "failed_scenes": failed_scenes,
            "time": time.time() - t0
        }

    # Publicación atómica
    tmp_out.replace(out_file)

    elapsed = time.time() - t0
    return {
        "status": "OK",
        "day": day_str,
        "scenes": downloaded_scenes,
        "time": elapsed,
        "rows": len(_WORKER_GRID_DF) * downloaded_scenes,
        "size_mb": out_file.stat().st_size / (1024 * 1024),
        "failed_scenes": []
    }


def resolve_grid_file(user_path: str = None) -> Path:
    """Busca inteligentemente cordoba_grid_2km_regions.parquet sin obligar a usar data/ready/."""
    candidates = []
    if user_path:
        p = Path(user_path)
        if p.exists() and p.is_file():
            return p.resolve()
        candidates.append(p)

    candidates.append(Path(__file__).resolve().parent / "cordoba_grid_2km_regions.parquet")
    candidates.append(Path.cwd() / "cordoba_grid_2km_regions.parquet")
    candidates.append(Path.cwd() / "data" / "cordoba_grid_2km_regions.parquet")
    candidates.append(Path.cwd() / "data" / "ready" / "cordoba_grid_2km_regions.parquet")
    candidates.append(ROOT_DIR / "data" / "ready" / "cordoba_grid_2km_regions.parquet")
    candidates.append(ROOT_DIR / "cordoba_grid_2km_regions.parquet")

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()

    return candidates[0] if candidates else Path("cordoba_grid_2km_regions.parquet")


def check_environment(grid_file: Path, out_dir: Path, tmp_dir: Path):
    """Verifica todos los requisitos del entorno para asegurar ejecución inmediata."""
    print("=" * 80)
    print("🔍 VERIFICACIÓN DE ENTORNO Y PORTABILIDAD")
    print("=" * 80)

    # 1. Librerías
    required = ["boto3", "pyarrow", "pyproj", "pvlib", "xarray", "h5netcdf", "pandas", "numpy"]
    all_ok = True
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg:12s}: Instalado")
        except ImportError:
            print(f"  ❌ {pkg:12s}: NO INSTALADO")
            all_ok = False

    # 2. Conectividad NOAA S3 (GOES-16 y GOES-19)
    try:
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        resp16 = s3.list_objects_v2(Bucket="noaa-goes16", Prefix="ABI-L1b-RadF/2021/018/14/", MaxKeys=1)
        resp19 = s3.list_objects_v2(Bucket="noaa-goes19", Prefix="ABI-L1b-RadF/2025/100/15/", MaxKeys=1)
        if "Contents" in resp16 and "Contents" in resp19:
            print("  ✓ Conexión S3 : NOAA GOES-16 y GOES-19 públicos accesibles (sin credenciales)")
        elif "Contents" in resp16:
            print("  ✓ Conexión S3 : NOAA GOES-16 accesible (GOES-19 sin respuesta)")
        else:
            print("  ⚠️ Conexión S3 : No devolvió objetos de prueba")
    except Exception as e:
        print(f"  ❌ Conexión S3 : Falló ({e})")
        all_ok = False

    # 3. Grilla 2 km
    resolved_grid = resolve_grid_file(grid_file)
    if resolved_grid.exists():
        size_mb = resolved_grid.stat().st_size / (1024 * 1024)
        print(f"  ✓ Grilla 2 km : Encontrada en {resolved_grid} ({size_mb:.2f} MB)")
    else:
        print(f"  ❌ Grilla 2 km : No encontrada (se buscó en . y subcarpetas)")
        all_ok = False

    # 4. Espacio en disco
    try:
        total, used, free = shutil.disk_usage(str(out_dir.parent if out_dir.parent.exists() else "."))
        free_gb = free / (1024**3)
        print(f"  ✓ Disco libre : {free_gb:.1f} GB disponibles")
        if free_gb < 10.0:
            print("  ⚠️ Alerta     : Menos de 10 GB disponibles en disco.")
    except Exception:
        pass

    print("=" * 80)
    if all_ok:
        print("✅ Entorno 100% verificado y listo para descargar.")
    else:
        print("❌ Por favor instale los paquetes faltantes o corrija las rutas indicadas.")
    print("=" * 80)
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Motor de Extracción Masiva Nativa 2 km GOES-16 (Portátil y Multinodo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de Uso Multinodo / Multimáquina:

1. Computadora 1 (Años 2020 a 2022):
   python3 scripts/fase_1_b_mass_grid2km.py --start 2020-01-01 --end 2022-12-31

2. Computadora 2 (Años 2023 a 2025):
   python3 scripts/fase_1_b_mass_grid2km.py --start 2023-01-01 --end 2025-11-30

3. Sharding balanceado (2 máquinas en paralelo sobre todo el período):
   PC 1: python3 scripts/fase_1_b_mass_grid2km.py --shard 0 --total-shards 2
   PC 2: python3 scripts/fase_1_b_mass_grid2km.py --shard 1 --total-shards 2

4. En una máquina nueva (sin índice local de S3):
   python3 scripts/fase_1_b_mass_grid2km.py --grid-file cordoba_grid_2km_regions.parquet --years 2023,2024
        """
    )
    parser.add_argument("--start", default="2020-01-01", help="Fecha inicio YYYY-MM-DD (def: 2020-01-01)")
    parser.add_argument("--end", default="2025-11-30", help="Fecha fin YYYY-MM-DD (def: 2025-11-30)")
    parser.add_argument("--years", default=None, help="Años específicos separados por coma (ej: 2020,2021,2022)")
    parser.add_argument("--days", default=None, help="Lista de días YYYYMMDD separados por coma o archivo .txt")
    parser.add_argument("--shard", type=int, default=0, help="ID de shard para ejecución distribuida (0-indexed)")
    parser.add_argument("--total-shards", type=int, default=1, help="Total de shards distribuidos (def: 1)")
    parser.add_argument("--workers", type=int, default=4, help="Número de workers concurrentes (def: 4)")
    parser.add_argument("--concurrency-per-file", type=int, default=4, help="Hilos multipart por archivo (def: 4)")
    parser.add_argument("--minute-step", type=int, default=30, choices=[10, 30, 60], help="Paso temporal en minutos")
    parser.add_argument("--grid-file", default=None, help="Ruta al archivo parquet de la grilla 2 km (auto-detectado si está en la misma carpeta)")
    parser.add_argument("--outdir", default=str(DEFAULT_OUT_DIR), help="Directorio destino Parquet 2 km")
    parser.add_argument("--tmpdir", default=str(DEFAULT_TMP_DIR), help="Directorio temporal de descarga")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Directorio opcional de índices locales")
    parser.add_argument("--log-file", default=str(DEFAULT_FAILED_LOG), help="Bitácora de escenas fallidas")
    parser.add_argument("--max-days", type=int, default=None, help="Límite opcional de días para pruebas")
    parser.add_argument("--check-env", action="store_true", help="Verifica el entorno de ejecución y sale")
    args = parser.parse_args()

    grid_path = resolve_grid_file(args.grid_file)
    out_dir = Path(args.outdir)
    tmp_dir = Path(args.tmpdir)
    index_dir = Path(args.index_dir) if args.index_dir else None
    log_file = Path(args.log_file)

    if args.check_env:
        ok = check_environment(grid_path, out_dir, tmp_dir)
        sys.exit(0 if ok else 1)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if not grid_path.exists():
        print(f"❌ ERROR: No se encontró la grilla de 2 km en '{grid_path}'.")
        print("   Por favor coloque cordoba_grid_2km_regions.parquet en la misma carpeta o use --grid-file <ruta>.")
        sys.exit(1)

    # Construir lista de días a procesar
    if args.days is not None:
        if Path(args.days).exists():
            with open(args.days) as f:
                raw_days = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        else:
            raw_days = [d.strip() for d in args.days.split(",") if d.strip()]
        days_to_process = sorted(list(set(raw_days)))
    elif args.years is not None:
        target_years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
        days_to_process = []
        for y in target_years:
            y_start = datetime(y, 1, 1)
            y_end = min(datetime(y, 12, 31), datetime(2025, 11, 30))
            cur = y_start
            while cur <= y_end:
                days_to_process.append(cur.strftime("%Y%m%d"))
                cur += timedelta(days=1)
    else:
        start_dt = pd.to_datetime(args.start)
        end_dt = pd.to_datetime(args.end)
        cur = start_dt
        days_to_process = []
        while cur <= end_dt:
            days_to_process.append(cur.strftime("%Y%m%d"))
            cur += timedelta(days=1)

    # Aplicar Sharding si total_shards > 1
    if args.total_shards > 1:
        if args.shard < 0 or args.shard >= args.total_shards:
            raise ValueError(f"Shard {args.shard} inválido para total_shards {args.total_shards}")
        days_to_process = [d for i, d in enumerate(days_to_process) if i % args.total_shards == args.shard]
        shard_info = f"Shard {args.shard + 1}/{args.total_shards}"
    else:
        shard_info = "Único (sin sharding)"

    print("=" * 80)
    print("🚀 MOTOR DE EXTRACCIÓN MASIVA NATIVA 2 KM GOES-16 (Convenio ATLAS II)")
    print(f"   • Días configurados: {len(days_to_process):,} ({days_to_process[0] if days_to_process else '-'} -> {days_to_process[-1] if days_to_process else '-'})")
    print(f"   • Modo de partición: {shard_info}")
    print(f"   • Workers concurrentes: {args.workers} (× {args.concurrency_per_file} threads = {args.workers * args.concurrency_per_file} conexiones)")
    print(f"   • Resolución temporal: cada {args.minute_step} min (muestreo solar adaptativo)")
    print(f"   • Grilla objetivo: {grid_path}")
    print(f"   • Directorio destino: {out_dir}")
    print(f"   • Directorio temporal: {tmp_dir}")
    print(f"   • Descubrimiento de escenas: {'Local + S3 on-the-fly' if index_dir and index_dir.exists() else '100% S3 on-the-fly (sin índices locales)'}")
    print(f"   • Bitácora de fallos: {log_file}")
    print(f"   • Compresión: ZSTD Level 3 (streaming incremental ParquetWriter)")
    print("=" * 80)

    existing_days = set()
    for f in out_dir.glob("reflectancia_*.parquet"):
        if f.stat().st_size > 1024 * 1024:
            existing_days.add(f.stem.replace("reflectancia_", ""))

    pending_days = [d for d in days_to_process if d not in existing_days]
    print(f"📊 Estado inicial: {len(existing_days):,} días ya completados en destino | {len(pending_days):,} días pendientes en este lote.")

    if args.max_days is not None and args.max_days > 0:
        pending_days = pending_days[:args.max_days]
        print(f"⚠️ Limitando corrida a {len(pending_days)} días solicitados.")

    if not pending_days:
        print("✅ Todos los días especificados ya se encuentran procesados. Nada que hacer.")
        return

    def cleanup_signal(sig, frame):
        print("\n🛑 Señal de interrupción recibida. Limpiando archivos temporales...")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        print("✓ Limpieza completa. Saliendo de forma segura.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_signal)
    signal.signal(signal.SIGTERM, cleanup_signal)

    t_global = time.time()
    completed_days = 0
    total_scenes = 0
    partial_days = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(grid_path, args.concurrency_per_file)
    ) as executor:
        futures = {
            executor.submit(
                process_single_day_worker,
                day_str,
                out_dir,
                tmp_dir,
                args.minute_step,
                index_dir
            ): day_str
            for day_str in pending_days
        }

        for future in as_completed(futures):
            day_str = futures[future]

            try:
                res = future.result()
            except Exception as exc:
                print(f"🔥 ERROR CATASTRÓFICO EN WORKER ({day_str}): {exc}")
                with open(log_file, "a") as f_log:
                    f_log.write(json.dumps({
                        "day": day_str,
                        "type": "WORKER_CRASH",
                        "error": str(exc),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }) + "\n")
                continue

            status = res["status"]

            if status == "OK":
                completed_days += 1
                total_scenes += res["scenes"]
                elapsed = time.time() - t_global
                rate_days = completed_days / (elapsed / 3600.0) if elapsed > 0 else 0
                eta_hours = (len(pending_days) - completed_days) / rate_days if rate_days > 0 else 0
                print(
                    f"[{completed_days:4d}/{len(pending_days):4d}] {day_str} | "
                    f"OK ({res['scenes']} esc, {res['size_mb']:.1f}MB, {res['time']:.1f}s) | "
                    f"Vel: {rate_days:4.1f} d/h | ETA: {eta_hours:4.1f} h"
                )
            elif status == "PARTIAL":
                partial_days += 1
                print(f"⚠️ {day_str}: INCOMPLETO ({res['downloaded']}/{res['expected']} escenas). Descartado.")
                with open(log_file, "a") as f_log:
                    for fs in res["failed_scenes"]:
                        f_log.write(json.dumps({
                            "day": day_str,
                            "type": "FAILED_SCENE",
                            "url": fs.get("url"),
                            "scene_timestamp": fs.get("timestamp"),
                            "error": fs.get("error"),
                            "recorded_at": datetime.now(timezone.utc).isoformat()
                        }) + "\n")
            elif status == "SKIPPED":
                pass
            else:
                print(f"⚠️ {day_str}: {status}")

    total_time = time.time() - t_global
    print("=" * 80)
    print(f"🎉 SESIÓN DE EXTRACCIÓN CONCLUIDA:")
    print(f"   • Días completados al 100%: {completed_days:,}")
    print(f"   • Días parciales/descartados: {partial_days:,}")
    print(f"   • Escenas satelitales procesadas: {total_scenes:,}")
    print(f"   • Tiempo total transcurrido: {total_time/3600:.2f} horas")
    if completed_days > 0:
        print(f"   • Velocidad media lograda: {completed_days / (total_time/3600):.1f} días/hora")
    print("=" * 80)


if __name__ == "__main__":
    main()
