#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 1 — Descarga e indexado GOES ABI-L1b-RadF

Funciones:
- Descarga escenas por día o rango de fechas, horas y paso en minutos.
- Guarda archivos en: data/goes/YYYYMMDD/
- Crea/actualiza: data/goes/YYYYMMDD/_index.csv (orden cronológico por start)
- Genera geometría solar horaria: data/geometry/solar_geom_YYYYMMDD.parquet

Ejemplos:
  # Un día, todas las horas
  python scripts/fase_1.py --fecha 2020-01-15 --sat goes16 --canal C02

  # Un día, 11–22 UTC, cada 30 min
  python scripts/fase_1.py --fecha 2020-01-15 --sat goes16 --canal C02 --hours 11-22 --minute-step 30

  # Rango de fechas, 11–22 UTC, cada 30 min
  python scripts/fase_1.py --start 2024-09-17 --end 2024-12-31 --sat goes16 --canal C02 --hours 11-22 --minute-step 30

Notas:
- --source {auto|s3|local}: auto intenta S3 y si falla usa lo que haya local.
- Requiere: boto3, pandas, geopandas, pvlib, pyarrow (o fastparquet).
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np # type: ignore
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
import pvlib # type: ignore

import boto3 # type: ignore
from botocore import UNSIGNED # type: ignore
from botocore.config import Config # type: ignore
from botocore.exceptions import BotoCoreError, ClientError # type: ignore

# ----------------- Config por defecto -----------------
BUCKET_DEFAULT = "noaa-goes16"
PRODUCT_DEFAULT = "ABI-L1b-RadF"


# ----------------- Utilidades geom/solar -----------------
def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


def generar_geom_solar(grid: gpd.GeoDataFrame, fecha_ini: pd.Timestamp, fecha_fin: pd.Timestamp) -> pd.DataFrame:
    horas = pd.date_range(fecha_ini, fecha_fin, freq="1H", tz="UTC")
    gg = grid.copy()
    if not np.all(gg.geometry.geom_type == "Point"):
        gg["geometry"] = gg.geometry.centroid
    gg["lon"] = gg.geometry.x.astype(float)
    gg["lat"] = gg.geometry.y.astype(float)

    recs = []
    for _, r in gg[["cell_id", "lat", "lon"]].iterrows():
        sp = pvlib.solarposition.get_solarposition(horas, r.lat, r.lon)
        df = pd.DataFrame({
            "timestamp": horas,
            "cell_id": int(r.cell_id),
            "zenith": sp["zenith"].to_numpy(float),
            "azimuth": sp["azimuth"].to_numpy(float),
        })
        df["cosz"] = np.cos(np.radians(df["zenith"]))
        recs.append(df)

    return pd.concat(recs, ignore_index=True)


# ----------------- Parsing de nombres GOES -----------------
# Ej.: OR_ABI-L1b-RadF-M6C02_G16_s20200151500198_e20200151509506_c20200151509551.nc
ABI_NAME_RE = re.compile(
    r"OR_(?P<product>ABI-L1b-[A-Za-z0-9]+)-M6C(?P<chan>\d{2})_G(?P<goes>\d{2})_"
    r"s(?P<sy>\d{4})(?P<sdoy>\d{3})(?P<sh>\d{2})(?P<sm>\d{2})(?P<ss>\d{2})\d*_"
    r"e(?P<ey>\d{4})(?P<edoy>\d{3})(?P<eh>\d{2})(?P<em>\d{2})(?P<es>\d{2})\d*_"
    r"c(?P<cy>\d{4})(?P<cdoy>\d{3})(?P<ch>\d{2})(?P<cm>\d{2})(?P<cs>\d{2})\d*\.nc$"
)


def _ydoyhms_to_ts(y: str, doy: str, h: str, mi: str, s: str) -> pd.Timestamp:
    y, doy, h, mi, s = int(y), int(doy), int(h), int(mi), int(s) # type: ignore
    base = pd.Timestamp(y, 1, 1, tz="UTC") + pd.Timedelta(days=doy - 1) # type: ignore
    return base + pd.Timedelta(hours=h, minutes=mi, seconds=s)


def parse_times_from_name(fname: str) -> Optional[Dict]:
    m = ABI_NAME_RE.search(os.path.basename(fname))
    if not m:
        return None
    gd = m.groupdict()
    return {
        "start": _ydoyhms_to_ts(gd["sy"], gd["sdoy"], gd["sh"], gd["sm"], gd["ss"]),
        "end": _ydoyhms_to_ts(gd["ey"], gd["edoy"], gd["eh"], gd["em"], gd["es"]),
        "created": _ydoyhms_to_ts(gd["cy"], gd["cdoy"], gd["ch"], gd["cm"], gd["cs"]),
        "chan": gd["chan"], "product": gd["product"], "goes": gd["goes"]
    }


# ----------------- S3 helpers -----------------
def s3_client_unsigned():
    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED, region_name="us-east-1",
                      retries={"max_attempts": 5, "mode": "standard"})
    )


def list_by_prefix_minute(
    s3, bucket: str, product: str, ts: pd.Timestamp, band_num: int, goes_id: str, debug: bool = False
) -> List[Dict]:
    """
    Lista objetos para un minuto exacto usando un prefijo 'sYYYYDOYHHMM' (cubre cualquier segundo).
    """
    year = ts.strftime("%Y"); doy = ts.strftime("%j"); hh = ts.strftime("%H"); mm = ts.strftime("%M")
    prefix = f"{product}/{year}/{doy}/{hh}/OR_{product}-M6C{band_num:02d}_G{goes_id}_s{year}{doy}{hh}{mm}"
    if debug:
        print(f"  · S3 prefix(min): s3://{bucket}/{prefix}*")

    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
    except (BotoCoreError, ClientError) as e:
        print(f"  ! list_objects_v2 error: {e}")
        return []

    out = []
    for it in resp.get("Contents", []):
        key = it["Key"]
        t = parse_times_from_name(key)
        if not t:
            continue
        # filtro de sanidad por canal y sat (ya está en el prefix, pero reafirmamos)
        if f"-M6C{band_num:02d}_" not in key:  # canal
            continue
        if f"_G{int(goes_id):02d}_" not in key:  # sat
            continue
        out.append({
            "key": key,
            "size": int(it.get("Size", 0)),
            "etag": it.get("ETag", "").strip('"'),
            "start": t["start"], "end": t["end"], "created": t["created"], "chan": t["chan"]
        })

    out.sort(key=lambda d: (d["start"].value, d["created"].value))
    return out


def download_one(s3, bucket: str, item: Dict, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = os.path.basename(item["key"])
    dest = out_dir / fname
    if not dest.exists():
        s3.download_file(bucket, item["key"], str(dest))
    d2 = dict(item); d2["local_path"] = str(dest)
    return d2


# ----------------- Índice por día -----------------
def update_index_manifest(day_dir: Path, items: List[Dict], band: str, goes_id: str, product: str):
    idx_csv = day_dir / "_index.csv"
    rows: List[Dict] = []

    # Cargar existente si lo hay
    if idx_csv.exists():
        try:
            old = pd.read_csv(idx_csv, parse_dates=["start", "end", "created"])
            rows.extend(old.to_dict("records"))
        except Exception:
            pass

    # Agregar nuevos
    for it in items:
        rows.append({
            "filename": os.path.basename(it["key"]),
            "local_path": it.get("local_path", str(day_dir / os.path.basename(it["key"]))),
            "size": it["size"],
            "etag": it.get("etag", ""),
            "start": it["start"], "end": it["end"], "created": it["created"],
            "band": band, "goes": goes_id, "product": product
        })

    if not rows:
        return

    df = pd.DataFrame(rows).drop_duplicates(subset=["filename"]).sort_values("start")
    df.to_csv(idx_csv, index=False)


# ----------------- Helpers CLI -----------------
def parse_hours(hours_str: str) -> List[int]:
    hours_str = hours_str.strip()
    if not hours_str:
        return list(range(24))
    if "-" in hours_str:
        h0, h1 = [int(x) for x in hours_str.split("-", 1)]
        return list(range(h0, h1 + 1))
    return [int(h) for h in hours_str.split(",") if h]


def day_range_from_args(fecha: Optional[str], start: Optional[str], end: Optional[str]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if fecha:
        d = pd.Timestamp(fecha, tz="UTC").floor("D")
        return d, d
    if start and end:
        s = pd.Timestamp(start, tz="UTC").floor("D")
        e = pd.Timestamp(end, tz="UTC").floor("D")
        if e < s:
            raise ValueError("--end debe ser >= --start")
        return s, e
    raise ValueError("Proveer --fecha o (--start y --end).")


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="Fase 1 — Descarga e indexado GOES ABI-L1b-RadF")
    # fechas
    ap.add_argument("--fecha", help="YYYY-MM-DD (opción simple de un día)")
    ap.add_argument("--start", help="YYYY-MM-DD (inicio, inclusive)")
    ap.add_argument("--end", help="YYYY-MM-DD (fin, inclusive)")
    # sat/producto/canal
    ap.add_argument("--sat", default="goes16", choices=["goes16", "goes18", "goes19"])
    ap.add_argument("--canal", default="C02", help="C01/C02/C03/…")
    ap.add_argument("--product", default=PRODUCT_DEFAULT)
    ap.add_argument("--bucket", default=BUCKET_DEFAULT)
    # tiempo
    ap.add_argument("--hours", default="", help="rango UTC ej. 11-22 o lista 11,12,13; vacío = 00..23")
    ap.add_argument("--minute-step", type=int, default=30, help="paso en minutos (ej 30)")
    ap.add_argument("--max-per-minute", type=int, default=1, help="máximo de objetos a bajar por minuto (1 usual)")
    # control
    ap.add_argument("--source", default="auto", choices=["auto", "s3", "local"],
                    help="auto: intenta S3 y si falla usa local; s3: solo S3; local: solo archivos locales")
    ap.add_argument("--debug-s3", action="store_true", help="imprime prefijos S3 por minuto")
    ap.add_argument("--overwrite", action="store_true", help="redescarga aunque exista")
    args = ap.parse_args()

    # Rango de días
    day_start, day_end = day_range_from_args(args.fecha, args.start, args.end)
    hours = parse_hours(args.hours)
    band_num = int(args.canal.replace("C", "").replace("c", ""))
    goes_id = {"goes16": "16", "goes18": "18", "goes19": "19"}[args.sat]

    # Estructura dirs
    root = Path(".")
    data_dir = root / "data"
    grids_dir = data_dir / "grids"
    geo_dir = data_dir / "geometry"
    goes_root = data_dir / "goes"
    meta_dir = Path("metadata")
    for d in [geo_dir, goes_root, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Rejilla obligatoria (para geometría solar)
    grid_file = grids_dir / "cordoba_grid_5km.geojson"
    if not grid_file.exists():
        raise FileNotFoundError(f"No existe {grid_file}. Corré fase_0.py primero.")
    grid = gpd.read_file(grid_file)[["cell_id", "geometry"]].copy()
    grid = ensure_wgs84(grid) # type: ignore
    print(f"🗺️  Rejilla cargada ({len(grid)} celdas).")

    # Cliente S3 si corresponde
    s3 = s3_client_unsigned() if args.source in ("auto", "s3") else None

    # Loop de días
    for day in pd.date_range(day_start, day_end, freq="D"):
        day_dir = goes_root / day.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        # Geometría solar horaria del día
        fecha_ini = day.floor("D"); fecha_fin = fecha_ini + pd.Timedelta(hours=23)
        print(f"🌞 Geometría solar (UTC) para {fecha_ini.date()}...")
        geo = generar_geom_solar(grid, fecha_ini, fecha_fin)
        geo_file = geo_dir / f"solar_geom_{fecha_ini.strftime('%Y%m%d')}.parquet"
        geo.to_parquet(geo_file)
        print(f"✅ Geometría solar guardada en {geo_file}")

        # Descarga por minuto
        new_items: List[Dict] = []
        if args.source in ("auto", "s3") and s3 is not None:
            print(f"🛰️ Descargando {args.product} {args.sat} banda {band_num:02d} para {day.date()} (UTC {hours[0]}–{hours[-1]} cada {args.minute_step} min)…")
            for hh in hours:
                for mm in range(0, 60, args.minute_step):
                    ts = (day + pd.Timedelta(hours=hh, minutes=mm)).tz_convert("UTC")
                    items = list_by_prefix_minute(s3, args.bucket, args.product, ts, band_num, goes_id, debug=args.debug_s3)
                    if not items:
                        continue
                    for it in items[: args.max_per_minute]:
                        fname = os.path.basename(it["key"])
                        dest = day_dir / fname
                        if dest.exists() and not args.overwrite:
                            d2 = dict(it); d2["local_path"] = str(dest)
                            new_items.append(d2)
                        else:
                            try:
                                d2 = download_one(s3, args.bucket, it, day_dir)
                                new_items.append(d2)
                                print(f"  ↓ {fname}")
                            except (BotoCoreError, ClientError) as e:
                                print(f"  ! error {e}")

        # Si no bajamos nada y se permite local, indexar lo que haya en disco
        if not new_items and args.source in ("auto", "local"):
            print("🔁 Usando archivos locales si existen…")
            local = []
            for p in sorted(day_dir.glob("*.nc")):
                t = parse_times_from_name(p.name)
                if not t:
                    continue
                if f"-M6C{band_num:02d}_" not in p.name:   # canal
                    continue
                if f"_G{int(goes_id):02d}_" not in p.name: # sat
                    continue
                local.append({
                    "key": p.name, "size": p.stat().st_size, "etag": "",
                    "start": t["start"], "end": t["end"], "created": t["created"],
                    "chan": t["chan"], "local_path": str(p)
                })
            local.sort(key=lambda d: (d["start"].value, d["created"].value))
            new_items = local
            if not new_items:
                print("  (sin archivos locales)")

        # Actualizar índice del día
        if new_items:
            update_index_manifest(day_dir, new_items, f"C{band_num:02d}", goes_id, args.product)
            print(f"🧭 _index.csv actualizado ({len(new_items)} escenas nuevas/registradas)")
        else:
            print("🧭 _index.csv sin cambios (no hubo escenas)")

    print("🎉 Fase 1 completada (descarga + índices + geometría solar).")


if __name__ == "__main__":
    main()
