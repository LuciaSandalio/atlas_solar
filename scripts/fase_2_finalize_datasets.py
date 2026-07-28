#!/usr/import/env python3
# -*- coding: utf-8 -*-

"""
Fase 2.7: Consolidación y Filtrado de los Datasets de Radiación Finales
Filtra el dataset completo para incluir únicamente las 166 estaciones aprobadas
(queda = 1.0, 0.5, 0.7, 0.2) y descarta las 79 estaciones marcadas con queda = 0.0.
"""

import os
import json
import pandas as pd
from pathlib import Path

# Paths
META_FILE = Path("data/raw/aws/resumen_estaciones_sol_todas.csv")
CALIBRATED_IN = Path("data/processed/stations/davis_calibrated_hourly.parquet")
PARAMS_IN = Path("data/processed/stations/davis_calibration_params.json")

CALIBRATED_OUT = Path("data/processed/stations/davis_calibrated_hourly_final.parquet")
PARAMS_OUT = Path("data/processed/stations/davis_calibration_params_final.json")

def main():
    print("🚀 Consolidando base de datos final del Atlas Solar de Córdoba...")
    
    if not META_FILE.exists() or not CALIBRATED_IN.exists() or not PARAMS_IN.exists():
        print("❌ Error: Faltan archivos de entrada en el espacio de trabajo.")
        return
        
    # 1. Obtener la lista de estaciones aprobadas
    meta_df = pd.read_csv(META_FILE)
    meta_df['queda_num'] = meta_df['queda'].astype(str).str.replace(',', '.').astype(float)
    
    # Estaciones aprobadas (queda = 1.0, 0.5, 0.7, 0.2)
    approved_stations = set(meta_df[meta_df['queda_num'].isin([1.0, 0.5, 0.7, 0.2])]['estacion'])
    print(f"📋 Estaciones aprobadas en los metadatos: {len(approved_stations)} de {len(meta_df)} totales.")
    
    # 2. Filtrar dataset calibrado Parquet
    print("📉 Cargando dataset calibrado completo...")
    df_cal = pd.read_parquet(CALIBRATED_IN)
    initial_stations = set(df_cal['estacion'].unique())
    print(f"   Estaciones en el parquet completo: {len(initial_stations)} | Registros: {len(df_cal)}")
    
    # Filtrar
    df_final = df_cal[df_cal['estacion'].isin(approved_stations)].copy()
    final_stations = set(df_final['estacion'].unique())
    print(f"   Estaciones filtradas retenidas: {len(final_stations)} | Registros: {len(df_final)}")
    
    # 2.5 Corregir desfasajes horarios (Timezone Shifts) dinámicamente
    print("🕒 Aplicando correcciones de husos horarios (shifts dinámicos)...")
    df_final = df_final.reset_index()
    
    # Asegurarnos de que `tiempo_utc` es dt
    df_final['tiempo_utc'] = pd.to_datetime(df_final['tiempo_utc'])
    
    shifted_count = 0
    for s in final_stations:
        mask = df_final['estacion'] == s
        df_s = df_final[mask]
        if not df_s.empty:
            envelope = df_s.groupby(df_s['tiempo_utc'].dt.hour)['solar_radiation'].quantile(0.90)
            peak_hour = envelope.idxmax()
            shift_hours = 16 - peak_hour
            if shift_hours != 0:
                df_final.loc[mask, 'tiempo_utc'] = df_final.loc[mask, 'tiempo_utc'] + pd.Timedelta(hours=shift_hours)
                shifted_count += 1
                
    df_final = df_final.set_index('tiempo_utc')
    print(f"   ✅ Correcciones dinámicas aplicadas a {shifted_count} estaciones.")
    
    # Guardar Parquet Final
    print("💾 Guardando dataset calibrado de radiación final en Parquet...")
    df_final.to_parquet(CALIBRATED_OUT, index=True)
    print(f"   ✅ ¡Parquet final guardado en {CALIBRATED_OUT}!")
    
    # 3. Filtrar parámetros de calibración JSON
    with open(PARAMS_IN) as f:
        params = json.load(f)
        
    final_params = {k: v for k, v in params.items() if k in approved_stations}
    print(f"📋 Coeficientes de calibración retenidos: {len(final_params)} estaciones.")
    
    # Guardar JSON Final
    with open(PARAMS_OUT, 'w') as f:
        json.dump(final_params, f, indent=2)
    print(f"   ✅ ¡JSON final guardado en {PARAMS_OUT}!")
    
    print("\n🎉 ¡Consolidación exitosa! Todos los datasets del Atlas Solar están listos.")

if __name__ == "__main__":
    main()
