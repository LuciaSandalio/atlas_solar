#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# Constants & Utils
# ------------------------------------------------------------------------------

DATA_DIR = Path("data")
REFLECTANCE_DIR = DATA_DIR / "processed"
GEOMETRY_DIR = DATA_DIR / "geometry"
OUTPUT_DIR = DATA_DIR / "ready" / "features_cell_hour"

def main():
    parser = argparse.ArgumentParser(description="Fase 1C: UnificaciÃ³n de Reflectancia y GeometrÃa Solar.")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD Start date")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD End date")
    parser.add_argument("--outdir", default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    start_date = pd.to_datetime(args.start).normalize()
    end_date = pd.to_datetime(args.end).normalize()
    
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Iniciando unificación desde {start_date.date()} hasta {end_date.date()}")

    current_date = start_date
    while current_date <= end_date:
        day_str = current_date.strftime("%Y%m%d")
        day_iso = current_date.strftime("%Y-%m-%d")

        # 1. Define paths
        ref_path = REFLECTANCE_DIR / f"reflectancia_{day_str}.parquet"
        geom_path = GEOMETRY_DIR / f"solar_geom_{day_str}.parquet"
        
        # 2. Check existence
        if not ref_path.exists():
            print(f"⚠️  Falta Reflectancia para {day_iso} ({ref_path}), saltando...")
            current_date += timedelta(days=1)
            continue
            
        if not geom_path.exists():
            print(f"⚠️  Falta Geometría para {day_iso} ({geom_path}), saltando...")
            current_date += timedelta(days=1)
            continue

        try:
            # 3. Load Data
            df_ref = pd.read_parquet(ref_path)
            df_geom = pd.read_parquet(geom_path)

            # Ensure timestamps are UTC
            if df_ref["timestamp"].dt.tz is None:
                df_ref["timestamp"] = df_ref["timestamp"].dt.tz_localize("UTC")
            else:
                df_ref["timestamp"] = df_ref["timestamp"].dt.tz_convert("UTC")

            if df_geom["timestamp"].dt.tz is None:
                df_geom["timestamp"] = df_geom["timestamp"].dt.tz_localize("UTC")
            else:
                df_geom["timestamp"] = df_geom["timestamp"].dt.tz_convert("UTC")

            # 4. Prepare for Merge
            # Check columns in geometry
            if "zenith" in df_geom.columns and "zenith_deg" not in df_geom.columns:
                df_geom.rename(columns={"zenith": "zenith_deg"}, inplace=True)
            
            if "zenith_deg" in df_geom.columns and "cosZ" not in df_geom.columns:
                 df_geom["cosZ"] = np.cos(np.deg2rad(df_geom["zenith_deg"]))

            # Round timestamps to nearest hour
            df_ref["time_round"] = df_ref["timestamp"].dt.round("1h")
            
            # Group by cell_id and rounded hour to get hourly averages
            df_ref_hourly = df_ref.groupby(["cell_id", "time_round"], as_index=False).agg({
                "reflectance": "mean",
                "lon": "first",
                "lat": "first"
            })
            
            df_geom["time_round"] = df_geom["timestamp"].dt.round("1h")
            df_geom_hourly = df_geom.drop_duplicates(subset=["cell_id", "time_round"])
            
            # 5. Merge
            cols_to_merge = ['cell_id', 'time_round', 'zenith_deg', 'cosZ']
            if 'azimuth' in df_geom_hourly.columns:
                cols_to_merge.append('azimuth')

            merged = pd.merge(
                df_ref_hourly,
                df_geom_hourly[cols_to_merge],
                on=["cell_id", "time_round"],
                how="left"
            )
            
            # 6. Save
            out_file = out_dir / f"features_{day_str}.parquet"
            merged.to_parquet(out_file, index=False)
            print(f"✅ {day_iso}: Guardado {out_file} ({len(merged)} filas)")

        except Exception as e:
            print(f"❌ Error procesando {day_iso}: {e}")

        current_date += timedelta(days=1)

    print("🎉 Fase 1C completada.")

if __name__ == "__main__":
    main()
