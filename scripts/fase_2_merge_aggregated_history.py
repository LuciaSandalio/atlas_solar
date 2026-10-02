#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fase 2.13: Fusionador Canónico de Historial Davis con Corrección Semántica de Huso Horario
==========================================================================================
1. Carga los archivos históricos de `data/raw/aws/hourly_parquet_agregados/` (y Villa Valeria),
   cuyos timestamps originales son naive y representan Hora Local Argentina (America/Argentina/Cordoba).
2. Convierte explícitamente a UTC real:
   ts.dt.tz_localize("America/Argentina/Cordoba").dt.tz_convert("UTC")
3. Carga la fuente moderna desde `hourly_parquet_backup/` (garantizando datos prístinos en UTC nativo).
4. Asigna trazabilidad de procedencia:
   - source = 'historical_aggregated'
   - source = 'modern_api'
5. En la frontera de solapamiento (finales de octubre 2020), aplica precedencia estricta:
   modern_api > historical_aggregated
6. Sobrescribe los archivos finales en `data/raw/aws/hourly_parquet/` asegurando continuidad temporal física.
"""

import os
import shutil
import glob
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw/aws/hourly_parquet"
AGREGADOS_DIR = BASE_DIR / "raw/aws/hourly_parquet_agregados"
BACKUP_DIR = BASE_DIR / "raw/aws/hourly_parquet_backup"
VILLA_VALERIA_AGREG = BASE_DIR / "raw/aws/villa_valeria_bioagroindustriacba.parquet"


def merge_single_station(ag_path, raw_path, backup_path):
    """
    Fusiona un archivo agregado histórico con su correspondiente archivo crudo de API moderna,
    aplicando conversión semántica de huso horario y precedencia modern_api > historical_aggregated.
    """
    # 1. Asegurar backup prístino
    if not backup_path.exists():
        if raw_path.exists():
            shutil.copy2(raw_path, backup_path)
        else:
            raise FileNotFoundError(f"No existe ni raw ni backup para {raw_path.name}")

    # 2. Cargar modern_api desde el backup prístino
    df_raw = pd.read_parquet(backup_path)
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        if "tiempo_utc" in df_raw.columns:
            df_raw["tiempo_utc"] = pd.to_datetime(df_raw["tiempo_utc"], utc=True, errors="coerce")
        else:
            raise ValueError(f"No se encontró tiempo_utc en {backup_path.name}")
    else:
        df_raw = df_raw.reset_index()
        df_raw["tiempo_utc"] = pd.to_datetime(df_raw["tiempo_utc"], utc=True, errors="coerce")

    df_raw["source"] = "modern_api"
    original_len = len(df_raw)

    # 3. Cargar y convertir históricamente desde hora local a UTC
    df_agreg = pd.read_parquet(ag_path)
    if "tiempo_utc" in df_agreg.columns:
        ts = pd.to_datetime(df_agreg["tiempo_utc"], errors="coerce")
    else:
        ts = pd.to_datetime(df_agreg.index, errors="coerce")
        df_agreg = df_agreg.reset_index()

    # Conversión semántica explícita
    df_agreg["tiempo_original"] = ts
    # Localizar como hora argentina y convertir a UTC (+3h respecto a UTC)
    if ts.dt.tz is None:
        df_agreg["tiempo_utc"] = ts.dt.tz_localize("America/Argentina/Cordoba").dt.tz_convert("UTC")
    else:
        df_agreg["tiempo_utc"] = ts.dt.tz_convert("UTC")

    df_agreg["source"] = "historical_aggregated"

    # 4. Fusión con precedencia: modern_api primero, luego drop_duplicates
    # Al poner df_raw primero y keep='first', cualquier timestamp solapado es retenido de modern_api
    common_cols = [c for c in df_raw.columns if c in df_agreg.columns]
    df_combined = pd.concat([df_raw[common_cols], df_agreg[common_cols]], ignore_index=True)
    df_combined = df_combined.dropna(subset=["tiempo_utc"])
    
    # Remover duplicados priorizando modern_api
    df_combined = df_combined.drop_duplicates(subset=["tiempo_utc"], keep="first")
    df_combined = df_combined.sort_values("tiempo_utc").set_index("tiempo_utc")

    # 5. Sobrescribir en RAW_DIR
    df_combined.to_parquet(raw_path)
    added_rows = len(df_combined) - original_len
    return original_len, len(df_combined), added_rows


def main():
    print("=" * 80)
    print("🚀 FASE 2.13: FUSIÓN CANÓNICA DE HISTORIAL DAVIS (CONVERSIÓN LOCAL -> UTC)")
    print("=" * 80)

    if not RAW_DIR.exists():
        print(f"❌ Error: No se encuentra {RAW_DIR}")
        return
    if not AGREGADOS_DIR.exists():
        print(f"❌ Error: No se encuentra {AGREGADOS_DIR}")
        return
    if not BACKUP_DIR.exists():
        print(f"❌ Error: No se encuentra {BACKUP_DIR}")
        return

    agregados_files = sorted(list(AGREGADOS_DIR.glob("*.parquet")))
    print(f"📂 Encontrados {len(agregados_files)} archivos históricos en '{AGREGADOS_DIR.name}'.")

    merged_count = 0
    skipped_count = 0
    total_added_rows = 0

    for ag_path in agregados_files:
        base_name = ag_path.stem

        raw_pattern = str(RAW_DIR / f"*_{base_name}_solar_radiation.parquet")
        matching_raw_files = glob.glob(raw_pattern)

        if not matching_raw_files:
            print(f"   ⚠️  Saltando {base_name}: Sin archivo crudo en {RAW_DIR.name}")
            skipped_count += 1
            continue

        raw_path = Path(matching_raw_files[0])
        backup_path = BACKUP_DIR / raw_path.name

        try:
            orig_len, new_len, added = merge_single_station(ag_path, raw_path, backup_path)
            print(f"   ✓ {base_name:<40} | Base: {orig_len:6d} | Fusionado: {new_len:6d} | +{added:5d} h agregadas")
            merged_count += 1
            total_added_rows += added
        except Exception as e:
            print(f"   ❌ Error en {base_name}: {e}")

    # Procesar Villa Valeria
    if VILLA_VALERIA_AGREG.exists():
        vv_raw = RAW_DIR / "531_villa_valeria_bioagroindustriacba_solar_radiation.parquet"
        vv_backup = BACKUP_DIR / vv_raw.name
        if vv_backup.exists() or vv_raw.exists():
            try:
                orig_len, new_len, added = merge_single_station(VILLA_VALERIA_AGREG, vv_raw, vv_backup)
                print(f"   ✓ {'villa_valeria (especial)':<40} | Base: {orig_len:6d} | Fusionado: {new_len:6d} | +{added:5d} h agregadas")
                merged_count += 1
                total_added_rows += added
            except Exception as e:
                print(f"   ❌ Error en villa_valeria: {e}")

    print("\n" + "=" * 80)
    print("📋 RESUMEN DE LA FUSIÓN CANÓNICA")
    print("=" * 80)
    print(f"• Estaciones fusionadas con éxito: {merged_count}")
    print(f"• Total de horas históricas corregidas a UTC: {total_added_rows:,}")
    print("• Semántica aplicada: America/Argentina/Cordoba -> UTC (+3h) con precedencia modern_api")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
