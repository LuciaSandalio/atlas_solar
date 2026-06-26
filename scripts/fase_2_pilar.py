#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

RAW_DIR = Path("data/raw/pilar")
OUT_DIR = Path("data/processed/stations")
OUT_FILE = OUT_DIR / "pilar_cmp21_hourly.parquet"


TIMEZONE_SRC = "Etc/GMT+3" # Posix style for GMT+3 is Etc/GMT+3 which is UTC-3
                           # Or better: construct naive and subtract 3 hours manually if needed,
                           # but pandas handling of 'America/Argentina/Cordoba' is safer if available.
                           # Let's use simple timedelta for robust "UTC-3" assumption without DST issues if SMN doesn't use DST.
                           # SMN usually stays on ART (UTC-3) all year round (no DST since 2009).

def main():
    print(f"🚀 Iniciando procesamiento de estación Pilar CMP21...")
    
    # 1. List files
    files = sorted(list(RAW_DIR.glob("*.xlsx")))
    if not files:
        print(f"❌ No se encontraron archivos .xlsx en {RAW_DIR}")
        return

    print(f"📂 Encontrados {len(files)} archivos.")

    dfs = []
    
    for f in files:
        print(f"   Reading {f.name}...")
        try:
            # Read all cols to inspect
            df = pd.read_excel(f)
            cols = [c.upper().strip() for c in df.columns]
            df.columns = cols
            
            # Normalize variations
            rename_map = {
                "IRRAD (W/M2)": "RAD",
                "R.GLOB": "RAD",
                "SEÑAL[MV]": "CRUDO(mV)",
                "AÑO": "YEAR",
                "DIA": "DOY" # Day of Year
            }
            df.rename(columns=rename_map, inplace=True)
            
            # Case 1: Standard (FECHA, HORA, MINUTO, RAD)
            if "FECHA" in df.columns and "HORA" in df.columns and "MINUTO" in df.columns:
                df.dropna(subset=["FECHA", "HORA", "MINUTO", "RAD"], inplace=True)
                df["FECHA"] = pd.to_datetime(df["FECHA"])
                time_deltas = pd.to_timedelta(df['HORA'], unit='h') + pd.to_timedelta(df['MINUTO'], unit='m')
                df['timestamp_local'] = df['FECHA'] + time_deltas

            # Case 2: Old Format (YEAR, DOY, HORA as HHMM, RAD)
            elif "YEAR" in df.columns and "DOY" in df.columns and "HORA" in df.columns:
                 df.dropna(subset=["YEAR", "DOY", "HORA", "RAD"], inplace=True)
                 
                 # Parse HHMM
                 # HORA might be int (1754)
                 hhmm = df["HORA"].astype(str).str.zfill(4) # "1754"
                 hh = hhmm.str[:2].astype(int)
                 mm = hhmm.str[2:].astype(int)
                 
                 # Construct date from Year + DOY
                 # to_datetime("2015" + "068", format="%Y%j")
                 df["year_doy"] = df["YEAR"].astype(int).astype(str) + df["DOY"].astype(int).astype(str).str.zfill(3)
                 base_date = pd.to_datetime(df["year_doy"], format="%Y%j")
                 
                 df['timestamp_local'] = base_date + pd.to_timedelta(hh, unit='h') + pd.to_timedelta(mm, unit='m')

            # Case 3: Combined FECHA_HORA (2023+)
            elif "FECHA_HORA" in df.columns:
                 df.dropna(subset=["FECHA_HORA", "RAD"], inplace=True)
                 # Ensure datetime (strip potential whitespace)
                 df["timestamp_local"] = pd.to_datetime(df["FECHA_HORA"].astype(str).str.strip(), errors='coerce')
                 if df["timestamp_local"].isna().all():
                     print(f"   ⚠️  2023+ Warning: Todas las fechas fallaron parsing en {f.name}")
                 df.dropna(subset=["timestamp_local"], inplace=True)
            
            else:
                print(f"   ⚠️  Schema desconocido en {f.name}: {df.columns.tolist()}")
                continue
            
            if "RAD" not in df.columns:
                 print(f"   ⚠️  No se encontró columna de radiación en {f.name} (Columnas: {df.columns})")
                 continue

            print(f"   ✅ {f.name}: {len(df)} filas leídas.")
            dfs.append(df[['timestamp_local', 'RAD']])

            
        except Exception as e:
            print(f"   ❌ Error leyendo {f.name}: {e}")

    if not dfs:
        print("❌ No se pudo procesar ningún dataframe.")
        return

    # 2. Concat
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.sort_values("timestamp_local", inplace=True)
    
    # 3. Clean
    # Values < 0 -> 0
    full_df["RAD"] = full_df["RAD"].clip(lower=0)
    
    # 4. Convert to UTC
    # Assumption: Local Time = UTC - 3
    # So UTC = Local Time + 3
    full_df["timestamp"] = full_df["timestamp_local"] + pd.Timedelta(hours=3)
    full_df["timestamp"] = full_df["timestamp"].dt.tz_localize("UTC")
    
    # 5. Resample Hourly
    # "Left" labeled (e.g. 10:00 covers 10:00-10:59) is standard in some cases,
    # but for instantaneous irradiance averaging, center or start is key.
    # Solar datasets usually use "beginning of interval" or "center". 
    # Let's align with the project plan: "Resamplear los datos de 1 minuto a promedios horarios [HH:00–HH:59] -> HH"
    # This implies labeling with the starting hour.
    
    full_df.set_index("timestamp", inplace=True)
    
    hourly = full_df["RAD"].resample("1h", label="left").mean()
    
    # Convert back to DataFrame
    final_df = hourly.to_frame(name="GHI").reset_index()
    
    # Add n_obs count if needed?
    # Plan says: "2.1.3 Calcular número de observaciones válidas por hora (n_obs)" (for Omixom).
    # Good practice to have it here too.
    stats = full_df["RAD"].resample("1h", label="left").count()
    final_df["n_obs"] = stats.values
    
    # Filter hours with too few observations?
    # Plan 2.1.4: "Filtrar horas con menos del 80 % de observaciones".
    # 80% of 60 mins = 48 mins.
    valid_mask = final_df["n_obs"] >= 48
    final_df = final_df[valid_mask].copy()
    
    # Add station ID metadata column if useful, or just keep it simple.
    final_df["station_id"] = "PILAR_CMP21"

    # 6. Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUT_FILE, index=False)
    
    print(f"✅ Procesamiento finalizado.")
    print(f"   Archivo guardado: {OUT_FILE}")
    print(f"   Periodo: {final_df['timestamp'].min()} - {final_df['timestamp'].max()}")
    print(f"   Total horas válidas: {len(final_df)}")
    if len(final_df) > 0:
        print(f"   Ejemplo:\n{final_df.head()}")

if __name__ == "__main__":
    main()
