#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fase 2.13: Fusionador de Historial Davis
Respalda y fusiona los archivos de historial previo (*.parquet de hourly_parquet_agregados)
con los archivos de radiación solar cruda correspondientes en data/raw/aws/hourly_parquet.
Esto extiende las series de tiempo de 68 estaciones hacia los años 2018-2020.
"""

import os
import shutil
import glob
import pandas as pd
from pathlib import Path

# Paths
RAW_DIR = Path("data/raw/aws/hourly_parquet")
AGREGADOS_DIR = Path("data/raw/aws/hourly_parquet_agregados")
BACKUP_DIR = Path("data/raw/aws/hourly_parquet_backup")

def main():
    print("🚀 Iniciando Fusión de Historial Davis (Fase 2.13)...")
    
    if not RAW_DIR.exists():
        print(f"❌ Error: No se encuentra la carpeta de datos crudos {RAW_DIR}")
        return
    if not AGREGADOS_DIR.exists():
        print(f"❌ Error: No se encuentra la carpeta de datos agregados históricos {AGREGADOS_DIR}")
        return
        
    # 1. Crear carpeta de backup si no existe
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Directorio de respaldo preparado en: {BACKUP_DIR}")
    
    # 2. Listar archivos históricos agregados
    agregados_files = sorted(list(AGREGADOS_DIR.glob("*.parquet")))
    print(f"📂 Encontrados {len(agregados_files)} archivos históricos en '{AGREGADOS_DIR.name}'.")
    
    merged_count = 0
    skipped_count = 0
    total_added_rows = 0
    
    for ag_path in agregados_files:
        base_name = ag_path.stem  # Ej: 'acipro_santa_rosa_1'
        
        # Buscar archivo crudo correspondiente
        # Patrón: '*_<base_name>_solar_radiation.parquet'
        raw_pattern = str(RAW_DIR / f"*_{base_name}_solar_radiation.parquet")
        matching_raw_files = glob.glob(raw_pattern)
        
        if not matching_raw_files:
            print(f"   ⚠️  Saltando {base_name}: No se encontró archivo crudo correspondiente en {RAW_DIR.name}")
            skipped_count += 1
            continue
            
        raw_path = Path(matching_raw_files[0])
        print(f"   🔄 Fusionando {base_name}...")
        
        try:
            # A) Hacer backup del archivo crudo original antes de cualquier modificación
            backup_file_path = BACKUP_DIR / raw_path.name
            if not backup_file_path.exists():
                shutil.copy2(raw_path, backup_file_path)
            
            # B) Cargar datasets
            df_agreg = pd.read_parquet(ag_path)
            df_raw = pd.read_parquet(raw_path)
            
            original_len = len(df_raw)
            
            # C) Formatear y alinear tiempo en el historial (tiempo_utc)
            df_agreg['tiempo_utc'] = pd.to_datetime(df_agreg['tiempo_utc'])
            if df_agreg['tiempo_utc'].dt.tz is None:
                df_agreg['tiempo_utc'] = df_agreg['tiempo_utc'].dt.tz_localize('UTC')
            df_agreg = df_agreg.set_index('tiempo_utc')
            
            # D) Asegurar formato de tiempo en el crudo original (index)
            if df_raw.index.tz is None:
                df_raw.index = df_raw.index.tz_localize('UTC')
                
            # E) Concatenar y remover duplicados manteniendo el registro del archivo principal crudo en solapamientos
            df_combined = pd.concat([df_agreg, df_raw])
            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
            df_combined = df_combined.sort_index()
            
            added_rows = len(df_combined) - original_len
            total_added_rows += added_rows
            
            # F) Guardar archivo combinado sobreescribiendo el original en RAW_DIR
            df_combined.to_parquet(raw_path)
            
            print(f"      ✓ Éxito: {original_len} -> {len(df_combined)} filas | +{added_rows} horas agregadas | Rango: {df_combined.index.min()} a {df_combined.index.max()}")
            merged_count += 1
            
        except Exception as e:
            print(f"      ❌ Error al procesar {base_name}: {e}")
            
    print("\n" + "="*50)
    print("RESUMEN DE FUSIÓN HISTÓRICA")
    print("="*50)
    print(f"📊 Estaciones procesadas con éxito: {merged_count} de {len(agregados_files)}")
    print(f"⚠️ Estaciones omitidas (sin archivo crudo): {skipped_count}")
    print(f"📈 Total de horas históricas añadidas a la red: {total_added_rows} registros")
    print(f"🛡️ Todos los originales han sido respaldados en {BACKUP_DIR.name}/")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
