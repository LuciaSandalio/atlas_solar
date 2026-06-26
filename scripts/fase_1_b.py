#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import tempfile
from pathlib import Path
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", "Geometry is in a geographic CRS")

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import pvlib

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from urllib.parse import urlparse

try:
    import s3fs
    _HAS_S3FS = True
except Exception:
    _HAS_S3FS = False


# -------------------------------
# Constantes y Utils
# -------------------------------

ESUN_VIS = {
    "C01": 726.721,
    "C02": 663.274,
    "C03": 441.868,
}

def parse_s3_url(url: str):
    """Limpia y parsea URL S3, evitando duplicación de buckets."""
    p = urlparse(url)
    if p.scheme != "s3" or not p.netloc or not p.path:
        raise ValueError(f"URL S3 inválida: {url}")
    bucket = p.netloc
    key = p.path.lstrip("/")
    # Corrección para evitar 'bucket/bucket/key'
    if key.startswith(bucket + "/"):
        key = key[len(bucket) + 1 :]
    return bucket, key

def goes_var_name(ds: xr.Dataset) -> str:
    if "Rad" in ds.data_vars:
        return "Rad"
    if "CMI" in ds.data_vars:
        return "CMI"
    raise KeyError("No encuentro variable de radiancia 'Rad' ni 'CMI' en el dataset.")

def lonlat_to_scan_radians(ds, lon_deg, lat_deg):
    """Convierte Lat/Lon a ángulos de escaneo GOES (x, y) en radianes."""
    gip = None
    for cand in ["goes_imager_projection", "geospatial_lat_lon", "goes_imager_projection:0"]:
        if cand in ds.variables:
            gip = ds[cand]
            break
    if gip is None:
        for vname, v in ds.variables.items():
            if hasattr(v, "attrs"):
                if v.attrs.get("grid_mapping_name", "") == "geostationary":
                    gip = v
                    break
    if gip is None:
        raise KeyError("No se encontró la variable de proyección.")

    attrs = getattr(gip, "attrs", {})
    sweep = str(attrs.get("sweep_angle_axis", "x"))
    lon0 = float(attrs.get("longitude_of_projection_origin"))
    h = float(attrs.get("perspective_point_height"))
    a = float(attrs.get("semi_major_axis"))
    b = float(attrs.get("semi_minor_axis"))

    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon0 = np.deg2rad(lon0)

    e2 = 1.0 - (b * b) / (a * a)
    clat = np.cos(lat)
    slat = np.sin(lat)
    rn = a / np.sqrt(1.0 - e2 * (slat ** 2))

    x = rn * clat * np.cos(lon - lon0)
    y = rn * clat * np.sin(lon - lon0)
    z = rn * (1.0 - e2) * slat

    rx = h - x
    ry = -y
    rz = z

    d = (h * h) - (rx * rx + ry * ry + rz * rz)
    mask = d <= 0
    d = np.where(mask, np.nan, d)

    x_scan = np.arctan2(-ry, rx)
    y_scan = np.arcsin(rz / np.sqrt(rx * rx + ry * ry + rz * rz))

    if sweep.lower().startswith("y"):
        x_scan, y_scan = y_scan, x_scan

    return x_scan, y_scan

# -------------------------------
# Funciones de Optimización (Lazy & Crop)
# -------------------------------

def open_goes_dataset(path_or_url: str):
    """Abre el dataset en modo LAZY (sin cargar a RAM)."""
    # 1) s3fs
    try:
        if path_or_url.startswith("s3://"):
            import s3fs
            fs = s3fs.S3FileSystem(anon=True)
            fobj = fs.open(path_or_url, mode="rb") 
            ds = xr.open_dataset(fobj, engine="h5netcdf", chunks={})
            return ds, (lambda: None)
        elif path_or_url.startswith("http"):
            ds = xr.open_dataset(path_or_url, engine="h5netcdf", chunks={})
            return ds, (lambda: None)
    except Exception:
        pass

    # 2) fallback boto3 (descarga a /tmp)
    if path_or_url.startswith("s3://"):
        bucket, key = parse_s3_url(path_or_url)
        s3c = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        local = Path(tempfile.gettempdir()) / Path(key).name
        try:
            if not local.exists():
                s3c.download_file(bucket, key, str(local))
        except Exception as e:
            raise IOError(f"No pude descargar {path_or_url}: {e}")
        
        ds = xr.open_dataset(local, engine="h5netcdf", chunks={})
        return ds, (lambda: local.unlink(missing_ok=True))
    
    # 3) local
    ds = xr.open_dataset(path_or_url, engine="h5netcdf", chunks={})
    return ds, (lambda: None)

def crop_goes_to_grid(ds, grid_gpd, buffer=0.01):
    """
    Recorta el Dataset de GOES al Bounding Box de la grilla.
    Maneja ejes invertidos automáticamente.
    """
    minx, miny, maxx, maxy = grid_gpd.total_bounds
    corners_lon = [minx, maxx, minx, maxx]
    corners_lat = [miny, miny, maxy, maxy]
    
    x_rad, y_rad = lonlat_to_scan_radians(ds, corners_lon, corners_lat)
    
    valid = np.isfinite(x_rad) & np.isfinite(y_rad)
    if not np.any(valid):
        raise ValueError("La grilla está fuera del disco visible.")
        
    x_valid = x_rad[valid]
    y_valid = y_rad[valid]
    
    x_min, x_max = x_valid.min() - buffer, x_valid.max() + buffer
    y_min, y_max = y_valid.min() - buffer, y_valid.max() + buffer

    # Detectar dirección del eje y crear slices correctos (min->max o max->min)
    if ds.x[0] < ds.x[-1]: 
        x_slice = slice(x_min, x_max)
    else: 
        x_slice = slice(x_max, x_min)
        
    if ds.y[0] < ds.y[-1]: 
        y_slice = slice(y_min, y_max)
    else: 
        y_slice = slice(y_max, y_min)
    
    ds_subset = ds.sel(x=x_slice, y=y_slice)
    
    if ds_subset.x.size == 0 or ds_subset.y.size == 0:
        raise ValueError(f"Recorte vacío. Revisa coordenadas.")
        
    return ds_subset

def get_radiance_da(ds: xr.Dataset) -> xr.DataArray:
    """Obtiene el DataArray crudo (sin cargar)."""
    vname = goes_var_name(ds)
    return ds[vname]

def apply_scale_offset(da: xr.DataArray) -> xr.DataArray:
    """Aplica escala/offset y convierte a float32."""
    attrs = da.attrs or {}
    fv = attrs.get("_FillValue", None)
    vr = attrs.get("valid_range", None)
    sf = attrs.get("scale_factor", None)
    ao = attrs.get("add_offset", None)

    data = da.astype("float32")
    if fv is not None:
        data = data.where(data != fv)
    if vr is not None and hasattr(vr, "__len__") and len(vr) == 2:
        data = data.where((data >= float(vr[0])) & (data <= float(vr[1])))

    if sf is not None and ao is not None:
        try:
            sf = float(sf)
            ao = float(ao)
            data = data * sf + ao
        except Exception:
            pass
    return data

def maybe_smooth_x(rad_da: xr.DataArray, k: int) -> xr.DataArray:
    if not k or k < 2:
        return rad_da
    if k % 2 == 0:
        k += 1
    return rad_da.rolling(x=k, center=True, min_periods=1).mean()

def compute_reflectance_from_radiance(L: np.ndarray, zenith_deg: np.ndarray, band: str) -> np.ndarray:
    Esun = ESUN_VIS.get(band.upper(), 663.274)
    cosz = np.cos(np.deg2rad(zenith_deg))
    rho = np.where(cosz > 0, (np.pi * L) / (Esun * cosz), np.nan)
    rho = np.clip(rho, 0.0, 1.2)
    return rho.astype("float32")

# -------------------------------
# Lógica de Negocio
# -------------------------------

def read_index_for_day(day: pd.Timestamp, band: str, sat: str, base="data/goes") -> pd.DataFrame:
    p = Path(base) / day.strftime("%Y%m%d") / "_index.csv"
    if not p.exists():
        raise FileNotFoundError(f"No existe índice: {p}")
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    
    url_col = cols.get("url")
    if url_col:
        df["url"] = df[url_col].astype(str).str.strip()
    else:
        bcol = cols.get("s3_bucket") or cols.get("bucket")
        kcol = cols.get("s3_key") or cols.get("key")
        if kcol:
            key_series = df[kcol].astype(str).str.strip()
            bucket_series = df[bcol].astype(str).str.strip() if bcol else "noaa-goes16"
            df["url"] = "s3://" + bucket_series + "/" + key_series
        else:
            raise ValueError("Índice inválido.")

    if "band" in df.columns:
        df = df[df["band"].astype(str).str.upper() == band.upper()]
    if "goes" in df.columns:
        df = df[df["goes"].astype(str).str.contains(sat[-2:])]

    # Parsing robusto de timestamp desde URL
    import re
    def parse_from_key(u):
        m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", u)
        if not m:
            return pd.NaT
        yyyy, jjj, hh, mm, ss = map(int, m.groups())
        dt_str = f"{yyyy}{jjj:03d}{hh:02d}{mm:02d}{ss:02d}"
        return pd.to_datetime(dt_str, format="%Y%j%H%M%S", utc=True, errors='coerce')
    
    df["timestamp"] = pd.to_datetime([parse_from_key(u) for u in df["url"]], utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    
    # Filtrar solo el día pedido
    day_start = day.normalize()
    day_end = day_start + timedelta(days=1)
    df = df[df["timestamp"].between(day_start, day_end, inclusive="left")]
    return df

def sample_scene(
    ds: xr.Dataset,
    grid: gpd.GeoDataFrame,
    scene_time_utc: pd.Timestamp,
    *,
    band: str,
    interp_method: str = "linear",
    smooth_x: int = 0,
) -> pd.DataFrame:
    
    g = grid 
    if "4326" not in str(g.crs):
        g = g.to_crs(4326)
        
    lon = g.geometry.centroid.x.to_numpy()
    lat = g.geometry.centroid.y.to_numpy()
    cell_id = g["cell_id"].to_numpy() if "cell_id" in g.columns else np.arange(len(g), dtype=int)

    # 1. Calcular coordenadas destino
    x_scan, y_scan = lonlat_to_scan_radians(ds, lon, lat)
    
    # 2. OPTIMIZACIÓN: Recortar dataset (memoria baja)
    ds_subset = crop_goes_to_grid(ds, g, buffer=0.005)
    
    # 3. Cargar recorte a RAM
    ds_subset.load()
    
    # 4. Procesar Radiancia
    rad_subset = get_radiance_da(ds_subset)
    rad_decoded = apply_scale_offset(rad_subset)
    rad_s = maybe_smooth_x(rad_decoded, smooth_x)

    # 5. Interpolación
    method = "nearest" if interp_method == "nearest" else "linear"
    rad_pts = rad_s.interp(x=("points", x_scan), y=("points", y_scan), method=method)
    L = rad_pts.values.astype("float32")

    # 6. Geometría Solar (Incluye Azimut para el Plan de Trabajo)
    N_points = len(lon)
    time_series = pd.Series(scene_time_utc, index=range(N_points))
    sp = pvlib.solarposition.get_solarposition(time_series, lat, lon)
    zenith_deg = sp["zenith"].to_numpy(dtype="float32")
    azimuth_deg = sp["azimuth"].to_numpy(dtype="float32") # <-- AGREGADO PARA PLAN DE TRABAJO

    # 7. Reflectancia
    rho = compute_reflectance_from_radiance(L, zenith_deg, band)

    return pd.DataFrame(
        {
            "cell_id": cell_id,
            "lon": lon.astype("float32"),
            "lat": lat.astype("float32"),
            "radiance": L,
            "zenith_deg": zenith_deg,
            "azimuth_deg": azimuth_deg, # <-- AGREGADO
            "reflectance": rho,
            "timestamp": pd.to_datetime(scene_time_utc),
        }
    )

def process_single_day(day: pd.Timestamp, args, grid_path: str): # <-- CORREGIDO: recibe grid_path (str)
    """Worker function."""
    str_day = day.strftime('%Y-%m-%d')
    try:
        # --- NUEVO: Cargar Grilla AQUÍ para evitar Deadlock ---
        grid_gpd = gpd.read_file(grid_path)
        if grid_gpd.crs is None or "4326" not in str(grid_gpd.crs):
            grid_gpd = grid_gpd.to_crs(4326)
        # ----------------------------------------------------

        try:
            df_idx = read_index_for_day(day, args.canal, args.sat)
        except FileNotFoundError:
            return f"{str_day}: Sin índice"

        urls = df_idx["url"].tolist()
        times = df_idx["timestamp"].tolist()
        
        if not urls:
            return f"{str_day}: Índice vacío"

        out_base = Path(args.outdir) / day.strftime("%Y%m%d")
        out_base.mkdir(parents=True, exist_ok=True)

        parts = []
        
        for url, t in zip(urls, times):
            hhmmss = pd.to_datetime(t).strftime("%H%M%S")
            ds = None
            closer = None
            try:
                ds, closer = open_goes_dataset(url)
                
                df_s = sample_scene(
                    ds,
                    grid_gpd,
                    pd.to_datetime(t, utc=True),
                    band=args.canal,
                    interp_method=args.interp,
                    smooth_x=args.smooth_x,
                )

                if args.partitioned:
                    part_dir = out_base / hhmmss
                    part_dir.mkdir(parents=True, exist_ok=True)
                    out_part = part_dir / "part.parquet"
                    df_s.to_parquet(out_part, index=False)
                else:
                    parts.append(df_s)

            except Exception as e:
                print(f"❌ Error en {str_day} - {os.path.basename(url)}: {e}")
            finally:
                if closer:
                    try: closer() 
                    except: pass
                if ds: ds.close()

        if args.combine_day:
            final_df = pd.DataFrame()
            if args.partitioned:
                dfs = []
                for d in sorted(out_base.glob("[0-9]*")):
                    p = d / "part.parquet"
                    if p.exists():
                        dfs.append(pd.read_parquet(p))
                if dfs:
                    final_df = pd.concat(dfs, ignore_index=True)
            else:
                if parts:
                    final_df = pd.concat(parts, ignore_index=True)
            
            if not final_df.empty:
                out_all = Path(args.outdir) / f"reflectancia_{day.strftime('%Y%m%d')}.parquet"
                final_df.to_parquet(out_all, index=False)
                return f"{str_day}: OK ({len(final_df)} filas)"
            
        return f"{str_day}: OK (Particionado)"

    except Exception as e:
        print(f"🔥 CRASH día {str_day}: {e}")
        raise e

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Fase 1B Paralelo Optimizado.")
    
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--fecha", help="YYYY-MM-DD.")
    group.add_argument("--start", help="YYYY-MM-DD.")
    
    ap.add_argument("--end", help="YYYY-MM-DD.")
    ap.add_argument("--workers", type=int, default=1)
    
    ap.add_argument("--canal", default="C02")
    ap.add_argument("--sat", default="goes16")
    ap.add_argument("--grid", default="data/grids/cordoba_grid_5km.geojson")
    ap.add_argument("--from-index", action="store_true")
    ap.add_argument("--source", choices=["s3", "local"], default="s3")
    ap.add_argument("--partitioned", action="store_true")
    ap.add_argument("--combine-day", action="store_true")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--interp", choices=["nearest", "linear"], default="linear")
    ap.add_argument("--smooth-x", type=int, default=0)

    args = ap.parse_args()

    if args.fecha:
        start_date = end_date = pd.to_datetime(args.fecha).tz_localize("UTC")
    elif args.start and args.end:
        start_date = pd.to_datetime(args.start).tz_localize("UTC")
        end_date = pd.to_datetime(args.end).tz_localize("UTC")
        if start_date > end_date:
            sys.exit("Error: start > end")
    else:
        sys.exit("Error: Falta --end")

    # --- CORRECCIÓN DEADLOCK: No leemos la grilla aquí, solo validamos que exista ---
    if not os.path.exists(args.grid):
        sys.exit(f"❌ Error: No existe archivo de grilla: {args.grid}")
    # --------------------------------------------------------------------------------

    days_to_process = []
    curr = start_date
    while curr <= end_date:
        days_to_process.append(curr)
        curr += timedelta(days=1)

    print(f"🚀 Iniciando OPTIMIZADO con {args.workers} workers para {len(days_to_process)} días.")
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            # Pasamos 'args.grid' (path string) en vez de objeto
            executor.submit(process_single_day, day, args, args.grid): day 
            for day in days_to_process
        }

        for future in as_completed(futures):
            day_ref = futures[future]
            try:
                res = future.result()
                print(f"✅ {res}")
            except Exception as exc:
                print(f"💥 Error crítico en {day_ref.date()}: {exc}")

    print("\n🏁 Procesamiento finalizado.")

if __name__ == "__main__":
    main()