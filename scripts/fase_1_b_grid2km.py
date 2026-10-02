#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 1B Grid 2km: Extracción Canónica de Radiancia y Reflectancia GOES-16
========================================================================
Extrae las observaciones satelitales directamente sobre la grilla oficial
de 2 km de la Provincia de Córdoba (59,837 celdas) utilizando la proyección
geodésica oficial de PROJ / pyproj sin distorsiones espaciales.

Proyección GOES-16 ABI Fixed Grid:
+proj=geos +h=35786023.0 +a=6378137.0 +b=6356752.31414 +lon_0=-75.0 +sweep=x +no_defs
"""

import os
import sys
import time
import argparse
import re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr
import pyproj
import pvlib

ROOT_DIR = Path(__file__).resolve().parent.parent
GRID_2KM_FILE = ROOT_DIR / "data" / "ready" / "cordoba_grid_2km_regions.parquet"
OUTPUT_DIR_2KM = ROOT_DIR / "data" / "processed_2km"

# Constante solar extraterrestre para Band 2 (C02, 0.64 um) en W/(m^2 um)
ESUN_C02 = 663.274


def get_scan_coordinates_2km(grid_df: pd.DataFrame, goes_ds: xr.Dataset = None):
    """
    Calcula los ángulos de escaneo canónicos (x, y) en radianes para las 59,837 celdas
    utilizando pyproj.Proj exacto. Se ejecuta en ~5 ms una sola vez en memoria.
    """
    if goes_ds is not None and "goes_imager_projection" in goes_ds:
        attrs = goes_ds["goes_imager_projection"].attrs
        h = float(attrs.get("perspective_point_height", 35786023.0))
        a = float(attrs.get("semi_major_axis", 6378137.0))
        b = float(attrs.get("semi_minor_axis", 6356752.31414))
        lon_0 = float(attrs.get("longitude_of_projection_origin", -75.0))
        sweep = str(attrs.get("sweep_angle_axis", "x"))
    else:
        h = 35786023.0
        a = 6378137.0
        b = 6356752.31414
        lon_0 = -75.0
        sweep = "x"

    p_geos = pyproj.Proj(proj="geos", h=h, a=a, b=b, lon_0=lon_0, sweep=sweep)
    lons = grid_df["lon"].values
    lats = grid_df["lat"].values

    X, Y = p_geos(lons, lats)
    x_scan = (X / h).astype(np.float64)
    y_scan = (Y / h).astype(np.float64)

    return x_scan, y_scan, p_geos, h


def parse_timestamp_from_filename(filename: str) -> pd.Timestamp:
    """Extrae el timestamp UTC del nombre estándar de archivo GOES-16 L1b."""
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", filename)
    if not m:
        raise ValueError(f"No se pudo parsear timestamp del archivo: {filename}")
    yyyy, jjj, hh, mm, ss = map(int, m.groups())
    dt_str = f"{yyyy}{jjj:03d}{hh:02d}{mm:02d}{ss:02d}"
    return pd.to_datetime(dt_str, format="%Y%j%H%M%S", utc=True)


def extract_scene_2km(
    nc_path: Path,
    grid_df: pd.DataFrame,
    x_scan: np.ndarray,
    y_scan: np.ndarray,
    buffer_rad: float = 0.003
) -> pd.DataFrame:
    """
    Procesa una escena NetCDF completa en disco local:
    recorta a Córdoba, decodifica escala/offset, interpola bilinealmente a 59,837 celdas,
    calcula la posición solar y reflectancia.
    """
    t0 = time.time()
    scene_time = parse_timestamp_from_filename(nc_path.name)

    with xr.open_dataset(nc_path, engine="h5netcdf") as ds:
        # 1. Bounding box de Córdoba en radianes de escaneo
        x_min, x_max = float(x_scan.min() - buffer_rad), float(x_scan.max() + buffer_rad)
        y_min, y_max = float(y_scan.min() - buffer_rad), float(y_scan.max() + buffer_rad)

        # En GOES-16 ABI el eje y va de Norte (+y) a Sur (-y), por lo que es decreciente
        if ds.y[0] > ds.y[-1]:
            y_slice = slice(y_max, y_min)
        else:
            y_slice = slice(y_min, y_max)

        if ds.x[0] < ds.x[-1]:
            x_slice = slice(x_min, x_max)
        else:
            x_slice = slice(x_max, x_min)

        rad_crop = ds["Rad"].sel(x=x_slice, y=y_slice).load()

        # 2. Aplicar escala y offset
        attrs = rad_crop.attrs
        sf = float(attrs.get("scale_factor", 1.0))
        ao = float(attrs.get("add_offset", 0.0))
        fv = attrs.get("_FillValue", None)

        rad_float = rad_crop.astype(np.float32)
        if fv is not None:
            rad_float = rad_float.where(rad_float != fv)
        rad_decoded = rad_float * sf + ao

        # 3. Interpolación bilineal directa sobre las 59,837 posiciones de la grilla
        rad_pts = rad_decoded.interp(
            x=("points", x_scan),
            y=("points", y_scan),
            method="linear"
        ).values.astype(np.float32)

    # 4. Geometría solar exacta para todas las celdas
    lats = grid_df["lat"].values
    lons = grid_df["lon"].values
    n_pts = len(lats)
    t_series = pd.Series(scene_time, index=range(n_pts))

    sp = pvlib.solarposition.get_solarposition(t_series, lats, lons)
    zenith_deg = sp["zenith"].values.astype(np.float32)
    azimuth_deg = sp["azimuth"].values.astype(np.float32)

    # 5. Cálculo físico de reflectancia corregida
    cosz = np.cos(np.deg2rad(zenith_deg))
    reflectance = np.where(cosz > 0.0, (np.pi * rad_pts) / (ESUN_C02 * cosz), np.nan)
    reflectance = np.clip(reflectance, 0.0, 1.2).astype(np.float32)

    elapsed = time.time() - t0

    df_scene = pd.DataFrame({
        "cell_id": grid_df["cell_id"].values.astype(np.int32),
        "lon": lons.astype(np.float32),
        "lat": lats.astype(np.float32),
        "radiance": rad_pts,
        "zenith_deg": zenith_deg,
        "azimuth_deg": azimuth_deg,
        "reflectance": reflectance,
        "timestamp": scene_time
    })

    return df_scene, elapsed


def process_local_day(
    day_str: str,
    nc_files: list,
    output_dir: Path = OUTPUT_DIR_2KM
) -> Path:
    """
    Procesa un conjunto de archivos NetCDF locales correspondientes a un día,
    y guarda el Parquet consolidado nativo de 2 km.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"reflectancia_{day_str}.parquet"

    print("=" * 80)
    print(f"🚀 FASE 1B (2 KM): PROCESANDO DÍA {day_str} CON PROYECCIÓN CANÓNICA PYPROJ")
    print(f"   • Archivos locales encontrados: {len(nc_files)}")
    print(f"   • Destino de salida: {out_file}")
    print("=" * 80)

    # 1. Cargar grilla de 2 km
    grid_df = pd.read_parquet(GRID_2KM_FILE)
    print(f"📂 Grilla oficial cargada: {len(grid_df):,} celdas.")

    # 2. Precomputar coordenadas de escaneo
    x_scan, y_scan, p_geos, h = get_scan_coordinates_2km(grid_df)
    print(f"✓ Coordenadas canónicas de escaneo calculadas en pyproj:")
    print(f"  - x_scan: [{x_scan.min():.5f}, {x_scan.max():.5f}] rad")
    print(f"  - y_scan: [{y_scan.min():.5f}, {y_scan.max():.5f}] rad")

    # 3. Procesar escenas secuencialmente
    all_scenes = []
    t_start = time.time()

    for idx, nc_path in enumerate(sorted(nc_files)):
        df_scene, t_sc = extract_scene_2km(nc_path, grid_df, x_scan, y_scan)
        all_scenes.append(df_scene)
        ts_str = df_scene["timestamp"].iloc[0].strftime("%H:%M:%S")
        rad_mean = df_scene["radiance"].mean()
        ref_mean = np.nanmean(df_scene["reflectance"])
        print(f"   [{idx+1:2d}/{len(nc_files):2d}] {nc_path.name[:45]}... | UTC: {ts_str} | Rad: {rad_mean:6.1f} | Ref: {ref_mean:5.3f} | {t_sc:4.2f} s")

    # 4. Concatenar y guardar
    df_day = pd.concat(all_scenes, ignore_index=True)
    df_day.to_parquet(out_file, index=False, compression="snappy")

    total_time = time.time() - t_start
    print("-" * 80)
    print(f"✅ DÍA {day_str} COMPLETADO EXITOSAMENTE:")
    print(f"   • Total escenas: {len(nc_files)}")
    print(f"   • Total filas: {len(df_day):,} (59,837 celdas × {len(nc_files)} escenas)")
    print(f"   • Tiempo total: {total_time:.2f} s ({total_time / len(nc_files):.3f} s/escena)")
    print(f"   • Archivo guardado: {out_file} ({out_file.stat().st_size / (1024*1024):.1f} MB)")
    print("=" * 80)

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Extracción canónica nativa 2 km GOES-16")
    parser.add_argument("--day", default="20210115", help="Fecha YYYYMMDD")
    parser.add_argument("--source-dir", default=str(ROOT_DIR / "data" / ".s3cache"), help="Directorio con NetCDF locales")
    parser.add_argument("--outdir", default=str(OUTPUT_DIR_2KM), help="Directorio de salida")
    args = parser.parse_args()

    source_path = Path(args.source_dir)
    day_str = args.day
    d_obj = datetime.strptime(day_str, "%Y%m%d")
    doy = f"{d_obj.timetuple().tm_yday:03d}"

    pattern = f"*_s{day_str[:4]}{doy}*.nc"
    nc_files = sorted(list(source_path.glob(pattern)))

    if not nc_files:
        print(f"❌ No se encontraron archivos NetCDF para el día {day_str} en {source_path}")
        sys.exit(1)

    process_local_day(day_str, nc_files, Path(args.outdir))


if __name__ == "__main__":
    main()
