#!/usr/import/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import pvlib

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

BASE_DIR = Path("data")
AWS_RAW_DIR = BASE_DIR / "raw/aws/hourly_parquet"
META_FILE = BASE_DIR / "raw/aws/resumen_estaciones_sol_todas.csv"
OUT_DIR = BASE_DIR / "processed/stations"
OUT_FILE = OUT_DIR / "davis_qc_hourly.parquet"

# Constante solar aproximada
I_SC = 1361.0

def main():
    print("🚀 Iniciando Control de Calidad (QC) de la red AWS Davis...")
    
    # 1. Cargar metadatos
    if not META_FILE.exists():
        print(f"❌ Error: No se encuentra el archivo de metadatos {META_FILE}")
        return
        
    meta_df = pd.read_csv(META_FILE)
    # Crear diccionario de mapeo: estacion -> (lat, lon)
    # Filtramos estaciones sin coordenadas y eliminamos duplicados de índice, convirtiendo a float
    valid_meta = meta_df.dropna(subset=['lat', 'lon']).copy()
    valid_meta['lat'] = valid_meta['lat'].astype(str).str.replace(',', '.').astype(float)
    valid_meta['lon'] = valid_meta['lon'].astype(str).str.replace(',', '.').astype(float)
    valid_meta = valid_meta.drop_duplicates(subset=['estacion'])
    coord_map = valid_meta.set_index('estacion')[['lat', 'lon']].to_dict('index')
    
    print(f"📂 Metadatos cargados: {len(coord_map)} estaciones con coordenadas válidas de un total de {len(meta_df)}.")
    
    # 2. Listar archivos parquet
    parquet_files = sorted(list(AWS_RAW_DIR.glob("*_solar_radiation.parquet")))
    print(f"📂 Encontrados {len(parquet_files)} archivos horarios de radiación.")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_dfs = []
    stats = {"processed": 0, "skipped_no_coord": 0, "rows_in": 0, "rows_out": 0}
    
    for fpath in parquet_files:
        estacion = fpath.name.replace('_solar_radiation.parquet', '')
        
        if estacion not in coord_map:
            print(f"   ⚠️  Saltando {estacion}: No tiene coordenadas válidas en los metadatos.")
            stats["skipped_no_coord"] += 1
            continue
            
        lat = coord_map[estacion]['lat']
        lon = coord_map[estacion]['lon']
        
        try:
            df = pd.read_parquet(fpath)
            stats["rows_in"] += len(df)
            
            # Asegurar que el índice es temporal (si no lo es, intentar usar 'tiempo_utc' si existe)
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'tiempo_utc' in df.columns:
                    df.set_index('tiempo_utc', inplace=True)
                    df.index = pd.to_datetime(df.index)
                else:
                    print(f"   ❌ {estacion}: No se pudo determinar el índice temporal.")
                    continue
                    
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
                
            # Descartar filas con radiación nula absoluta (NaNs)
            df = df.dropna(subset=['solar_radiation'])
            
            # 3. Calcular posición solar
            solpos = pvlib.solarposition.get_solarposition(
                time=df.index,
                latitude=lat,
                longitude=lon
            )
            
            # 4. Aplicar Filtros de Control de Calidad Físico
            zenith = solpos['zenith']
            cos_z = np.cos(np.radians(zenith))
            
            # A) Filtro Nocturno: Descartar donde zenith >= 85 grados
            mask_day = zenith < 85.0
            
            # B) Filtro Físico Superior BSRN: GHI <= 1.2 * I_sc * cos(z)^1.2
            # BSRN Extremes limits usually use 1.2 * I_sc * cos(z)^1.2 + 50
            limit_upper = 1.2 * I_SC * np.power(np.maximum(cos_z, 0), 1.2) + 50.0
            mask_upper = df['solar_radiation'] <= limit_upper
            
            # C) Filtro Físico Inferior: GHI >= 0
            mask_lower = df['solar_radiation'] >= 0.0
            
            # D) Filtro de Valores Atascados (Stuck/Flatline Values)
            # Detecta si el sensor se queda congelado/colgado reportando el mismo valor constante
            # 1. Filtro general de persistencia de 24 horas (aplica a cualquier valor constante por 24 horas consecutivas)
            consec_counts = df.groupby((df['solar_radiation'] != df['solar_radiation'].shift()).cumsum())['solar_radiation'].transform('size')
            mask_stuck_general = consec_counts >= 24
            
            # 2. Filtro diurno (valores > 5.0 W/m2 constantes por 4 o más horas diurnas consecutivas)
            df_diurn_nonzero = df[(mask_day) & (df['solar_radiation'] > 5.0)].copy()
            if len(df_diurn_nonzero) > 0:
                consec_diurn = df_diurn_nonzero.groupby((df_diurn_nonzero['solar_radiation'] != df_diurn_nonzero['solar_radiation'].shift()).cumsum())['solar_radiation'].transform('size')
                stuck_diurn_times = df_diurn_nonzero[consec_diurn >= 4].index
                mask_stuck_diurn = df.index.isin(stuck_diurn_times)
            else:
                mask_stuck_diurn = False
                
            # 3. Filtro de Sensor Muerto (Dead Sensor / Extreme Low Max Filter)
            # Si el máximo valor de radiación diurna en todo un día es < 30.0 W/m2, es físicamente imposible
            # y representa un sensor roto, cubierto o reportando únicamente ruido eléctrico de base.
            df_daytime = df[mask_day].copy()
            if len(df_daytime) > 0:
                df_daytime['date'] = df_daytime.index.date
                daily_max = df_daytime.groupby('date')['solar_radiation'].transform('max')
                dead_daytime_times = df_daytime[daily_max < 30.0].index
                mask_dead_sensor = df.index.isin(dead_daytime_times)
            else:
                mask_dead_sensor = False
                
            mask_not_stuck = ~(mask_stuck_general | mask_stuck_diurn | mask_dead_sensor)
            
            # Combinar máscaras
            mask_valid = mask_day & mask_upper & mask_lower & mask_not_stuck
            
            df_qc = df[mask_valid].copy()
            
            if len(df_qc) > 0:
                # Añadir nombre de estación para el consolidado
                df_qc['estacion'] = estacion
                df_qc['latitude'] = lat
                df_qc['longitude'] = lon
                all_dfs.append(df_qc)
                
            stats["rows_out"] += len(df_qc)
            stats["processed"] += 1
            
        except Exception as e:
            print(f"   ❌ Error al procesar {fpath.name}: {e}")
            
    # Consolidación final
    if all_dfs:
        print("💾 Consolidando todos los datasets QC en un único parquet...")
        final_df = pd.concat(all_dfs)
        final_df.to_parquet(OUT_FILE, index=True)
        print(f"✅ ¡Completado! Dataset consolidado guardado en {OUT_FILE}.")
        print(f"📊 Estadísticas: {stats['processed']} archivos procesados, {stats['skipped_no_coord']} omitidos.")
        print(f"📊 Registros originales: {stats['rows_in']} | Registros válidos (QC): {stats['rows_out']} ({(stats['rows_out']/stats['rows_in']*100):.1f}%)")
    else:
        print("⚠️ No se generó ningún dato tras el QC.")

if __name__ == "__main__":
    main()
