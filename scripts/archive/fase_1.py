#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
import pvlib
import s3fs
import boto3
from botocore import UNSIGNED
from botocore.config import Config


# Paths base
DATA_DIR = "data"
GOES_DIR = os.path.join(DATA_DIR, "goes")
GEOM_DIR = os.path.join(DATA_DIR, "geometry")
META_DIR = "metadata"
os.makedirs(GOES_DIR, exist_ok=True)
os.makedirs(GEOM_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# -------------------- Utils --------------------

def ymd_to_doy(d: datetime) -> str:
    return f"{d.timetuple().tm_yday:03d}"

def parse_hours(hours_arg: str):
    # "11-22" -> range(11,23); "11,13,15" -> [11,13,15]
    if "-" in hours_arg:
        a, b = hours_arg.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(h) for h in hours_arg.split(",")]

def load_grid_lonlat(grid_path: str):
    """Centroides calculados en CRS proyectado para evitar warnings; salida en WGS84."""
    g = gpd.read_file(grid_path)
    if g.crs is None:
        g = g.set_crs(4326)
    if g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    g_m = g.to_crs(3857)
    c_m = g_m.geometry.centroid
    c = gpd.GeoSeries(c_m, crs=3857).to_crs(4326)
    lat = c.y.to_numpy()
    lon = c.x.to_numpy()
    return lat, lon, g.index.values

def calc_daylight_flag(day: datetime, grid_path: str, hours, minute_step: int):
    """Devuelve dict {HH: bool} indicando si en esa hora hay sol (cosZ>0.05 en alguna celda)."""
    lat, lon, _ = load_grid_lonlat(grid_path)
    # muestreo rápido (máximo 200 puntos)
    n = len(lat)
    if n > 200:
        idx = np.random.RandomState(42).choice(n, 200, replace=False)
        lat = lat[idx]; lon = lon[idx]; n = 200

    flags = {}
    for H in hours:
        ts = pd.date_range(
            start=pd.Timestamp(day.replace(hour=H, minute=0, second=0, microsecond=0), tz="UTC"),
            periods=max(1, 60 // minute_step),
            freq=f"{minute_step}min",
        )
        ok = False
        for t in ts:
            tvec = pd.DatetimeIndex([t] * n)  # misma longitud que lat/lon
            sp = pvlib.solarposition.get_solarposition(tvec, lat, lon)
            cosz = np.cos(np.deg2rad(sp["zenith"].to_numpy()))
            if np.nanmax(cosz) > 0.05:
                ok = True
                break
        flags[H] = ok
    return flags

def list_s3_hour(fs, bucket, product, year, doy, hour, debug=False):
    """Lista objetos en s3://{bucket}/{product}/{year}/{doy}/{hour:02d}/ con s3fs y fallback boto3 UNSIGNED."""
    prefix = f"{product}/{year}/{doy}/{hour:02d}/"
    s3path = f"{bucket}/{prefix}"
    urls = []
    # 1) s3fs
    try:
        lst = fs.ls(s3path)
        urls = [f"s3://{p}" for p in lst]
        if debug:
            print(f"    s3fs.ls('{s3path}') → {len(urls)} objetos")
    except Exception as e:
        if debug:
            print(f"    s3fs.ls('{s3path}') falló: {e}")

    # 2) boto3 fallback si quedó vacío
    if not urls:
        try:
            s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            contents = resp.get("Contents", [])
            urls = [f"s3://{bucket}/{c['Key']}" for c in contents]
            if debug:
                print(f"    boto3.list_objects_v2 Prefix='{prefix}' → {len(urls)} objetos")
        except Exception as e:
            if debug:
                print(f"    boto3.list_objects_v2 falló: {e}")
    return urls

def filter_m_band(urls, canal, sat):
    """Filtra por canal y satélite (independiente de M3/M6)."""
    want = []
    tag_c = f"C{int(canal[-2:]):02d}"
    tag_s = f"G{''.join([c for c in sat if c.isdigit()])}"
    for u in urls:
        name = os.path.basename(u)
        if tag_c in name and tag_s in name:
            want.append(u)
    return want

def write_index_for_day(day: datetime, rows):
    daystr = day.strftime("%Y%m%d")
    folder = os.path.join(GOES_DIR, daystr)
    os.makedirs(folder, exist_ok=True)
    idx = pd.DataFrame(rows, columns=["url", "start_iso", "band", "sat"])
    idx.to_csv(os.path.join(folder, "_index.csv"), index=False)
    return len(idx)

def write_geom_for_day(day: datetime, grid_path: str, minute_step: int):
    daystr = day.strftime("%Y%m%d")
    geom_out = os.path.join(GEOM_DIR, f"solar_geom_{daystr}.parquet")
    if os.path.exists(geom_out):
        return geom_out

    lat, lon, cell_ids = load_grid_lonlat(grid_path)
    ts = pd.date_range(
        start=pd.Timestamp(day.replace(hour=0, minute=0, second=0, microsecond=0), tz="UTC"),
        end=pd.Timestamp(day.replace(hour=23, minute=59, second=59, microsecond=0), tz="UTC"),
        freq=f"{minute_step}min",
        inclusive="both",
    )
    recs = []
    n = len(lat)
    for t in ts:
        tvec = pd.DatetimeIndex([t] * n)
        sp = pvlib.solarposition.get_solarposition(tvec, lat, lon)
        zen = sp["zenith"].to_numpy().astype("float32")
        cosz = np.cos(np.deg2rad(zen)).astype("float32")
        recs.append(pd.DataFrame({
            "cell_id": cell_ids,
            "timestamp": np.repeat(t, n),
            "zenith_deg": zen,
            "cosZ": cosz,
        }))
    df = pd.concat(recs, ignore_index=True)
    df.to_parquet(geom_out, index=False)
    return geom_out

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Fase 1: indexar escenas S3 y guardar geometría solar.")
    ap.add_argument("--fecha", help="YYYY-MM-DD (una fecha) o usar --start/--end")
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--sat", default="goes16")
    ap.add_argument("--canal", default="C02")
    ap.add_argument("--product", default="ABI-L1b-RadF")
    ap.add_argument("--bucket", default="noaa-goes16")
    ap.add_argument("--grid", default="data/grids/cordoba_grid_5km.geojson")
    ap.add_argument("--hours", default="11-22")
    ap.add_argument("--minute-step", type=int, default=30)
    ap.add_argument("--list-mode", default="s3fs", choices=["s3fs"])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-filter", action="store_true",
                    help="No filtra por canal/sat al indexar (modo diagnóstico)")
    args = ap.parse_args()

    if args.fecha:
        start = end = datetime.strptime(args.fecha, "%Y-%m-%d")
    else:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end   = datetime.strptime(args.end, "%Y-%m-%d")

    hours = parse_hours(args.hours)
    fs = s3fs.S3FileSystem(anon=True)

    d = start
    while d <= end:
        year = d.strftime("%Y")
        doy  = ymd_to_doy(d)
        daystr = d.strftime("%Y%m%d")

        # Prefiltro de luz diurna
        daylight = calc_daylight_flag(d, args.grid, hours, args.minute_step)

        rows = []
        print(f"🛰️ Indexando {args.product} {args.sat} banda {args.canal[-2:]} para {daystr} (UTC)…")
        for H in hours:
            if not daylight.get(H, True):
                print(f"  · {year}-{doy} {H:02d}Z: SALTEO por noche (cosZ<=0.05)")
                continue

            urls = list_s3_hour(fs, args.bucket, args.product, year, doy, H, debug=args.debug)
            if args.no_filter:
                urls_use = urls
            else:
                urls_use = filter_m_band(urls, args.canal, args.sat)

            if args.debug:
                print(f"  · {year}-{doy} {H:02d}Z: {len(urls)} objetos en hora; {len(urls_use)} tras filtro canal/sat")
                for name in [os.path.basename(u) for u in urls[:5]]:
                    print("    -", name)

            for u in urls_use:
                name = os.path.basename(u)
                # extraer start time del nombre …_sYYYYJJJHHMMSS…
                try:
                    sidx = name.find("_s")
                    
                    stamp = name[sidx + 2 : sidx + 2 + 13]  # YYYY (4) + JJJ (3) + HHMMSS (6) = 13
                    
                    t0 = pd.to_datetime(stamp, format="%Y%j%H%M%S", utc=True)
                except Exception:
                    # Fallback menos preciso si el nombre es irregular
                    t0 = pd.Timestamp(d.replace(hour=H, minute=0), tz="UTC")
                
                rows.append((u, t0.isoformat(), args.canal, args.sat))
            print(f"  · {year}-{doy} {H:02d}Z: {len(urls_use)} candidatos útiles")

        n = write_index_for_day(d, rows)
        print(f"🧭 Índice guardado: {os.path.join(GOES_DIR, daystr, '_index.csv')} ({n} filas)")

        geom_path = write_geom_for_day(d, args.grid, args.minute_step)
        print(f"🌞 Geometría solar guardada: {geom_path}")

        d += timedelta(days=1)

    print("🎉 Fase 1 completada (indexado S3 + geometría).")

if __name__ == "__main__":
    main()
