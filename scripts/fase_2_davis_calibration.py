#!/usr/import/env python3
# -*- coding: utf-8 -*-

"""
Fase 2.5: Calibración Virtual de la Red Davis usando REST2 Directo (2022-2025)
Ajusta la curva de corrección K(cosZ) para cada una de las 97 estaciones
utilizando sus marcas de tiempo y el modelo REST2 directo simulado.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import json

# Paths
INPUT_FILE = Path("data/processed/stations/davis_qc_with_rest2.parquet")
OUT_DIR = Path("data/processed/stations")
PARAMS_OUT = OUT_DIR / "davis_calibration_params.json"
CALIBRATED_OUT = OUT_DIR / "davis_calibrated_hourly.parquet"

def poly_fit_k(cosZ, c0, c1, c2):
    return c0 + c1 * cosZ + c2 * (cosZ ** 2)

def main():
    print("🚀 Iniciando la Calibración Virtual de la Red Davis...")
    
    if not INPUT_FILE.exists():
        print(f"❌ Error: No se encuentra el archivo {INPUT_FILE}")
        return
        
    print("📉 Cargando dataset con REST2 incorporado...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"   Registros diurnos totales: {len(df)} | Estaciones: {df['estacion'].nunique()}")
    
    # Asegurar que el índice está ordenado temporalmente por estación
    df = df.sort_index()
    
    calibration_params = {}
    DEFAULT_COEFFS = [1.0, 0.0, 0.0]
    
    # Para almacenar DataFrames calibrados
    calibrated_dfs = []
    
    print("\n⚡ Calibrando estaciones año por año (Calibración Segmentada)...")
    
    for est, df_est in df.groupby('estacion'):
        df_est = df_est.copy()
        
        # 1. Detectar cielo claro a nivel estación (para el fit global de fallback)
        df_est['k_raw'] = df_est['solar_radiation'] / df_est['ghi_cs']
        mask_basic = (
            (df_est['ghi_cs'] > 100.0) & 
            (df_est['zenith'] < 75.0) & 
            (df_est['k_raw'] >= 0.5) & 
            (df_est['k_raw'] <= 1.5)
        )
        rolling_std = df_est['k_raw'].rolling('3h', min_periods=3).std()
        mask_stable = rolling_std <= 0.05
        mask_clear = mask_basic & mask_stable
        df_clear = df_est[mask_clear].copy()
        
        # Filtro de outliers global
        if len(df_clear) >= 30:
            df_clear['K_target'] = df_clear['ghi_cs'] / df_clear['solar_radiation']
            q1 = df_clear['K_target'].quantile(0.25)
            q3 = df_clear['K_target'].quantile(0.75)
            iqr = q3 - q1
            mask_iqr = (df_clear['K_target'] >= q1 - 1.5 * iqr) & (df_clear['K_target'] <= q3 + 1.5 * iqr)
            df_clear_global = df_clear[mask_iqr].copy()
        else:
            df_clear_global = pd.DataFrame()

        # Ajuste global de la estación (fallback)
        c_global = DEFAULT_COEFFS
        global_status = "global_default"
        if len(df_clear_global) >= 30:
            try:
                popt_g, _ = curve_fit(
                    poly_fit_k, 
                    df_clear_global['cosZ'].values, 
                    df_clear_global['K_target'].values,
                    p0=[1.0, 0.0, 0.0],
                    bounds=([0.6, -1.5, -1.5], [1.8, 1.5, 1.5])
                )
                c_global = popt_g
                global_status = "global_calibrated"
            except Exception:
                pass
                
        # Estructura de parámetros para la estación
        calibration_params[est] = {
            "global": {
                "c0": float(c_global[0]), "c1": float(c_global[1]), "c2": float(c_global[2]),
                "n_clear_points": len(df_clear_global), "status": global_status
            },
            "yearly": {}
        }
        
        # 2. Calibración año por año
        df_est['ghi_original'] = df_est['solar_radiation']
        df_est['calibration_factor'] = 1.0
        
        calibrated_years = []
        # Agrupar por año (usando el año local de Córdoba)
        df_est['year'] = df_est.index.tz_convert('America/Argentina/Cordoba').year
        
        # Pre-convertir el índice de df_clear para extracciones anuales sin el sesgo del filtro IQR global
        if len(df_clear) > 0:
            df_clear_local = df_clear.copy()
            df_clear_local.index = df_clear_local.index.tz_convert('America/Argentina/Cordoba')
        else:
            df_clear_local = pd.DataFrame()

        for yr, df_yr in df_est.groupby('year'):
            df_yr = df_yr.copy()
            
            # Filtrar registros de cielo claro para este año específico (usando df_clear_local)
            if len(df_clear_local) > 0:
                df_clear_yr = df_clear_local[df_clear_local.index.year == yr].copy()
                # Aplicar filtro IQR local para este año
                if len(df_clear_yr) >= 30:
                    df_clear_yr['K_target'] = df_clear_yr['ghi_cs'] / df_clear_yr['solar_radiation']
                    q1 = df_clear_yr['K_target'].quantile(0.25)
                    q3 = df_clear_yr['K_target'].quantile(0.75)
                    iqr = q3 - q1
                    mask_iqr_yr = (df_clear_yr['K_target'] >= q1 - 1.5 * iqr) & (df_clear_yr['K_target'] <= q3 + 1.5 * iqr)
                    df_clear_yr = df_clear_yr[mask_iqr_yr].copy()
            else:
                df_clear_yr = pd.DataFrame()
            
            c_yr = c_global
            yr_status = "fallback_to_global"
            n_clear_yr = len(df_clear_yr)
            
            if n_clear_yr >= 10:
                try:
                    popt_yr, _ = curve_fit(
                        poly_fit_k, 
                        df_clear_yr['cosZ'].values, 
                        df_clear_yr['K_target'].values,
                        p0=[1.0, 0.0, 0.0],
                        bounds=([0.6, -1.5, -1.5], [1.8, 1.5, 1.5])
                    )
                    c0, c1, c2 = popt_yr
                    # Verificación física
                    test_val = poly_fit_k(0.5, c0, c1, c2)
                    if 0.6 <= test_val <= 1.8:
                        c_yr = popt_yr
                        yr_status = "yearly_calibrated"
                except Exception:
                    pass
            
            calibration_params[est]["yearly"][str(yr)] = {
                "c0": float(c_yr[0]), "c1": float(c_yr[1]), "c2": float(c_yr[2]),
                "n_clear_points": n_clear_yr, "status": yr_status
            }
            
            # Aplicar calibración de este año
            k_factors = poly_fit_k(df_yr['cosZ'].values, *c_yr)
            k_factors = np.clip(k_factors, 0.6, 1.8)
            df_yr['solar_radiation'] = df_yr['solar_radiation'] * k_factors
            df_yr['calibration_factor'] = k_factors
            
            calibrated_years.append(df_yr)
            
        if calibrated_years:
            df_cal_est = pd.concat(calibrated_years)
            
            # --- CONTROL DE CALIDAD TRIMESTRAL DINÁMICO (QC Físico) ---
            # Suprimimos warnings de timezone temporalmente
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_cal_est['quarter'] = df_cal_est.index.tz_convert('America/Argentina/Cordoba').to_period('Q')
            
            for q, df_q in df_cal_est.groupby('quarter'):
                active_cal = df_q.loc[df_q['solar_radiation'] > 50, 'solar_radiation']
                active_cs = df_q.loc[df_q['ghi_cs'] > 50, 'ghi_cs']
                
                if len(active_cal) > 50 and len(active_cs) > 50:
                    p99_cal = np.percentile(active_cal, 99.9)
                    p99_cs = np.percentile(active_cs, 99.9)
                    
                    # Criterio: ±25% del máximo teórico de cielo claro esperado para ese trimestre
                    if p99_cal > p99_cs * 1.25 or p99_cal < p99_cs * 0.75:
                        idx_q = df_q.index
                        df_cal_est.loc[idx_q, 'solar_radiation'] = np.nan
                        print(f"   ⚠️ QC Trimestral: {q} RECHAZADO (P99_cal={p99_cal:.1f}, P99_cs={p99_cs:.1f})")
            
            df_cal_est = df_cal_est.drop(columns=['quarter'])
            # ----------------------------------------------------------
            
            calibrated_dfs.append(df_cal_est)
            
            years_calibrated = [y for y, p in calibration_params[est]["yearly"].items() if p["status"] == "yearly_calibrated"]
            print(f"   ✅ {est}: Procesada. Años calibrados específicamente: {years_calibrated}")
            
    # Guardar parámetros
    with open(PARAMS_OUT, 'w') as f:
        json.dump(calibration_params, f, indent=2)
    print(f"\n💾 Coeficientes de calibración guardados en {PARAMS_OUT}")
    
    # Consolidar base final
    print("💾 Guardando base unificada calibrada...")
    final_df = pd.concat(calibrated_dfs)
    final_df.to_parquet(CALIBRATED_OUT, index=True)
    print(f"✅ ¡Calibración masiva finalizada exitosamente! Parquet guardado en {CALIBRATED_OUT}")

if __name__ == "__main__":
    main()
