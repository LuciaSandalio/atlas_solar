#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fase 3: Cálculo del Brillo Despejado (B0) (Arquitectura Big Data - 6 Años)
Ajusta la línea base de reflectancia en cielo claro para la provincia de Córdoba
utilizando muestreo estadístico y luego genera un dataset liviano de entrenamiento.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import json
import random

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/ready/b0_model")
STATIONS_FILE = Path("data/processed/stations/davis_calibrated_hourly_final.parquet")

# Ecuación de Tarpley clear-sky
def b0_model(X, A, B, C, D):
    cosZ, sinZ, cosGamma = X
    return A + B * cosZ + C * sinZ * cosGamma + D * sinZ * (cosGamma ** 2)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    files = sorted(list(INPUT_DIR.glob("reflectancia_*.parquet")))
    print(f"Archivos satelitales totales (Días): {len(files)}")
    
    if len(files) == 0:
        print("No se encontraron archivos en data/processed/")
        return
        
    # --- PASO 1: Muestreo Aleatorio para Modelo Físico B0 ---
    print("\n--- PASO 1: Ajuste del Modelo B0 (Muestreo Aleatorio) ---")
    
    # Tomar 72 días al azar de los 6 años (aprox 1 día por mes) para ajustar el B0
    random.seed(42)
    sample_files = random.sample(files, min(72, len(files)))
    
    df_sample_list = []
    for f in sample_files:
        df_sample_list.append(pd.read_parquet(f))
        
    df_sample = pd.concat(df_sample_list, ignore_index=True)
    
    df_sample['zenith_rad'] = np.deg2rad(df_sample['zenith_deg'])
    df_sample['cosZ'] = np.cos(df_sample['zenith_rad'])
    df_sample['sinZ'] = np.sin(df_sample['zenith_rad'])
    df_sample['cosGamma'] = np.cos(np.deg2rad(df_sample['azimuth_deg']))
    
    # Filtrar solo día claro
    df_sample = df_sample[df_sample['cosZ'] > 0.1].copy()
    
    df_sample['hour'] = df_sample['timestamp'].dt.hour
    
    print("Determinando horas despejadas (15% inferior de reflectancia)...")
    def get_clear_sky(group):
        q = group['reflectance'].quantile(0.15)
        return group[group['reflectance'] <= q]
    
    # DeprecationWarning para DataFrameGroupBy.apply se puede suprimir, pero usaremos el comportamiento default
    df_clear = df_sample.groupby(['cell_id', 'hour'], group_keys=False).apply(get_clear_sky).reset_index(drop=True)
    print(f"Pixeles despejados muestreados: {len(df_clear)}")
    
    X = (df_clear['cosZ'].values, df_clear['sinZ'].values, df_clear['cosGamma'].values)
    Y = df_clear['reflectance'].values
    
    print("Ajustando modelo BD-JPT / Tarpley B0...")
    popt, pcov = curve_fit(b0_model, X, Y, p0=[0.1, 0.1, 0.0, 0.0])
    
    model_params = {
        "A": popt[0],
        "B": popt[1],
        "C": popt[2],
        "D": popt[3],
        "MSE": float(np.mean((b0_model(X, *popt) - Y)**2))
    }
    
    print(json.dumps(model_params, indent=2))
    with open(OUTPUT_DIR / "B0_params.json", "w") as f:
        json.dump(model_params, f, indent=2)
        
    # --- PASO 2: Extracción Inteligente para Entrenamiento ---
    print("\n--- PASO 2: Extracción Selectiva de Features (Todas las Estaciones) ---")
    
    print("Cargando estaciones terrestres...")
    df_ground = pd.read_parquet(STATIONS_FILE)
    
    # Obtener un pixel de muestra para buscar las celdas más cercanas
    df_sample_cells = df_sample[['cell_id', 'lat', 'lon']].drop_duplicates()
    
    # df_ground format check
    if 'latitude' in df_ground.columns and 'longitude' in df_ground.columns:
        stations_unique = df_ground[['station_id', 'latitude', 'longitude']].drop_duplicates().dropna()
    else:
        # Check index if it's there
        if 'latitude' in df_ground.index.names:
            stations_unique = df_ground.reset_index()[['station_id', 'latitude', 'longitude']].drop_duplicates().dropna()

    def get_nearest_cell(lat, lon):
        dist = (df_sample_cells['lat'] - lat)**2 + (df_sample_cells['lon'] - lon)**2
        return df_sample_cells.loc[dist.idxmin(), 'cell_id']
        
    station_cells = set()
    for _, row in stations_unique.iterrows():
        station_cells.add(get_nearest_cell(row['latitude'], row['longitude']))
        
    print(f"Pixeles satelitales relevantes para entrenamiento: {len(station_cells)} celdas.")
    
    print(f"Procesando {len(files)} archivos para extracción (esto puede demorar unos minutos)...")
    
    output_rows = []
    
    for i, f in enumerate(files):
        if i % 100 == 0:
            print(f"Procesando {i}/{len(files)}...")
            
        df_day = pd.read_parquet(f)
        df_day = df_day[df_day['cell_id'].isin(station_cells)].copy()
        
        if len(df_day) == 0:
            continue
            
        df_day['zenith_rad'] = np.deg2rad(df_day['zenith_deg'])
        df_day['cosZ'] = np.cos(df_day['zenith_rad'])
        
        # Filtro de noche
        df_day = df_day[df_day['cosZ'] > 0.1].copy()
        
        if len(df_day) == 0:
            continue
            
        df_day['sinZ'] = np.sin(df_day['zenith_rad'])
        df_day['cosGamma'] = np.cos(np.deg2rad(df_day['azimuth_deg']))
        
        X_all = (df_day['cosZ'].values, df_day['sinZ'].values, df_day['cosGamma'].values)
        df_day['B0'] = b0_model(X_all, *popt)
        df_day['B0'] = df_day['B0'].clip(lower=0.01)
        
        # Ajustar columnas para mantener compatibilidad
        df_day['time_round'] = df_day['timestamp'].dt.round('h')
        
        # Quedarse con las columnas importantes para minimizar RAM
        cols_to_keep = ['cell_id', 'lat', 'lon', 'time_round', 'reflectance', 'cosZ', 'B0']
        output_rows.append(df_day[cols_to_keep])
        
    print("Concatenando features...")
    df_final = pd.concat(output_rows, ignore_index=True)
    print(f"Dataset de entrenamiento generado: {len(df_final)} filas a lo largo de 6 años.")
    
    print("Guardando station_features_B0.parquet...")
    df_final.to_parquet(OUTPUT_DIR / "station_features_B0.parquet", index=False)
    print("Fase 3 completada con éxito.")

if __name__ == "__main__":
    main()
