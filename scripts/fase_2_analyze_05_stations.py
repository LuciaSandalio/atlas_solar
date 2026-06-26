#!/usr/import/env python3
# -*- coding: utf-8 -*-

"""
Fase 2.6: Análisis Científico de Estaciones Davis con Queda = 0.5 (y 0.2, 0.7)
Evalúa la calidad física, calibración y longitud de serie temporal de cada estación dudosa.
Genera un informe con recomendaciones específicas de inclusión/exclusión.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
META_FILE = Path("data/raw/aws/resumen_estaciones_sol_todas.csv")
PARAMS_FILE = Path("data/processed/stations/davis_calibration_params.json")
CALIBRATED_FILE = Path("data/processed/stations/davis_calibrated_hourly.parquet")
REPORT_OUT = Path("data/processed/stations/davis_05_analysis_report.csv")

def main():
    print("🔬 Iniciando análisis de estaciones dudosas (Queda = 0.5, 0.7, 0.2)...")
    
    if not META_FILE.exists() or not PARAMS_FILE.exists() or not CALIBRATED_FILE.exists():
        print("❌ Error: Faltan archivos necesarios (meta, params o dataset calibrado).")
        return
        
    # Cargar metadatos
    meta_df = pd.read_csv(META_FILE)
    meta_df['queda_num'] = meta_df['queda'].astype(str).str.replace(',', '.').astype(float)
    
    # Filtrar dudosas
    df_dudosas = meta_df[meta_df['queda_num'].isin([0.5, 0.7, 0.2])].copy()
    print(f"📂 Encontradas {len(df_dudosas)} estaciones dudosas en los metadatos.")
    
    # Cargar parámetros de calibración
    with open(PARAMS_FILE) as f:
        params = json.load(f)
        
    # Cargar dataset calibrado completo (solo las dudosas para ahorrar RAM)
    print("📉 Cargando series de tiempo de estaciones dudosas...")
    df_cal = pd.read_parquet(CALIBRATED_FILE)
    
    results = []
    
    for idx, row in df_dudosas.iterrows():
        est = row['estacion']
        queda_orig = row['queda']
        queda_val = row['queda_num']
        
        # Filtrar datos de la estación en el dataset procesado
        df_est = df_cal[df_cal['estacion'] == est]
        
        if df_est.empty:
            results.append({
                "estacion": est,
                "queda_original": queda_orig,
                "n_records_qc": 0,
                "years_active": 0,
                "calibration_status": "no_data_in_parquet",
                "k_cosz_05": np.nan,
                "n_clear_points": 0,
                "recomienda": "NO",
                "motivo": "Sin datos válidos en el dataset procesado (QC total = 0)"
            })
            continue
            
        # Calcular estadísticas clave
        n_qc = len(df_est)
        
        # Años activos (años con al menos 200 horas válidas de radiación)
        df_est_yearly = df_est.groupby(df_est.index.year)['solar_radiation'].count()
        active_years = list(df_est_yearly[df_est_yearly >= 200].index)
        n_years = len(active_years)
        
        # Recuperar calibración
        est_params = params.get(est, {})
        global_params = est_params.get("global", {})
        cal_status = global_params.get("status", "unknown")
        
        c0 = global_params.get("c0", 1.0)
        c1 = global_params.get("c1", 0.0)
        c2 = global_params.get("c2", 0.0)
        n_clear = global_params.get("n_clear_points", 0)
        
        # Calcular K(cosZ = 0.5)
        k_05 = c0 + c1 * 0.5 + c2 * 0.25 if cal_status == "calibrated" else np.nan
        
        # Análisis de calidad:
        # - ¿Tiene una calibración estable? (k_05 entre 0.7 y 1.4)
        # - ¿Tiene al menos 1 año de datos válidos?
        # - ¿La radiación máxima registrada es físicamente realista (> 200 W/m2)?
        max_rad = df_est['solar_radiation'].max()
        
        recomienda = "SÍ"
        motivo = "Serie temporal estable, calibración virtual física y consistente."
        
        if cal_status != "calibrated":
            if n_clear < 10:
                recomienda = "NO"
                motivo = f"Falla de calibración: Muy pocas horas de cielo despejado (N_clear={n_clear}). Posible sensor tapado o dañado."
            else:
                recomienda = "ANALIZAR_DETALLE"
                motivo = f"Calibración virtual fallback. Requiere inspección visual."
        elif k_05 < 0.7 or k_05 > 1.4:
            recomienda = "NO"
            motivo = f"Calibración inestable / Sensor dañado o sucio (K(cosZ=0.5) = {k_05:.3f} fuera de límites)."
        elif n_years < 1:
            recomienda = "NO"
            motivo = f"Serie de tiempo extremadamente corta (< 1 año de datos activos)."
        elif max_rad < 300:
            recomienda = "NO"
            motivo = f"Lecturas extremadamente bajas (Máx GHI = {max_rad:.1f} W/m2). Sensor defectuoso o cubierto."
            
        results.append({
            "estacion": est,
            "queda_original": queda_orig,
            "n_records_qc": n_qc,
            "years_active": n_years,
            "years_list": str(active_years),
            "calibration_status": cal_status,
            "k_cosz_05": round(k_05, 3) if not np.isnan(k_05) else np.nan,
            "n_clear_points": n_clear,
            "recomienda": recomienda,
            "motivo": motivo
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(REPORT_OUT, index=False)
    print(f"\n📊 Reporte final de dudosas guardado en {REPORT_OUT}")
    
    print("\nResumen de Recomendaciones:")
    print(df_res['recomienda'].value_counts())

if __name__ == "__main__":
    main()
