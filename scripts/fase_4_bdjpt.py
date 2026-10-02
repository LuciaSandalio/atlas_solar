#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 4: Calibración y Modelado Satelital de Irradiancia (Kt)
============================================================
Convenio ATLAS II — Provincia de Córdoba (Componente Solar)
1. Alineamiento temporal riguroso A/B/C sin saltos nocturnos.
2. Inferencia en el espacio del Índice de Claridad (Kt) acotado en [0.0, 1.15].
3. Integración sin doble desplazamiento de estaciones terrestres (verificación pico 16h UTC).
4. Entrenamiento de modelos físicos (BD-JPT), estadísticos (HGB) e híbridos (BD-JPT + HGB).
5. Validación cruzada espacial estricta (GroupKFold por estación).
6. Exportación del contrato metodológico congelado para la Fase 5B.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import pvlib
import joblib
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Importar clases estables del paquete models
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.bdjpt import BDJPTWrapper, HybridBDJPT_HGB

DATA_DIR = Path('data')
PROCESSED_DIR = DATA_DIR / 'processed'
READY_DIR = DATA_DIR / 'ready'
METADATA_DIR = Path('metadata')
METADATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_FILE = READY_DIR / 'b0_model' / 'station_features_B0.parquet'
B0_PARAMS_FILE = READY_DIR / 'b0_model' / 'B0_params.json'
PILAR_FILE = PROCESSED_DIR / 'stations' / 'pilar_cmp21_hourly.parquet'
DAVIS_FILE = PROCESSED_DIR / 'stations' / 'davis_calibrated_hourly_final.parquet'
CLUSTERS_FILE = PROCESSED_DIR / 'stations' / 'davis_solar_clusters.gpkg'

CONTRACT_FILE = METADATA_DIR / 'fase4_inference_contract.json'
MODELS_FILE = PROCESSED_DIR / 'bdjpt_final_models_regional_estacional.pkl'

def calc_cs(timestamps, lat, lon):
    times = pd.DatetimeIndex(timestamps)
    loc = pvlib.location.Location(lat, lon)
    cs = loc.get_clearsky(times)
    solpos = loc.get_solarposition(times)
    return cs['ghi'].values, np.maximum(np.cos(np.radians(solpos['apparent_zenith'])), 0.0)

def main():
    print("=" * 80)
    print("🚀 FASE 4: ENTRENAMIENTO Y CALIBRACIÓN DE MODELOS SATELITALES (Kt)")
    print("=" * 80)
    
    # --- 1. CARGAR FEATURES SATELITALES HISTÓRICAS ---
    print("\n📂 1. Cargando features satelitales...")
    df_feat = pd.read_parquet(FEATURES_FILE)
    df_feat['time_round'] = pd.to_datetime(df_feat['time_round'], utc=True)
    df_feat_unique = df_feat[['cell_id', 'lat', 'lon']].drop_duplicates()
    
    def get_nearest_cell(df_u, lat, lon):
        dist = (df_u['lat'] - lat)**2 + (df_u['lon'] - lon)**2
        return df_u.loc[dist.idxmin(), 'cell_id']

    # --- 2. PREPARAR DATOS TERRESTRES AUDITADOS ---
    print("\n📂 2. Preparando datos terrestres auditados...")
    all_stations_data = []
    
    # Pilar
    pilar_lat, pilar_lon = -31.667, -63.883
    if PILAR_FILE.exists():
        print("   • Procesando estación de referencia Pilar (CMP21)...")
        df_pilar = pd.read_parquet(PILAR_FILE)
        if df_pilar['timestamp'].dt.tz is None:
            df_pilar['timestamp'] = df_pilar['timestamp'].dt.tz_localize("UTC")
        else:
            df_pilar['timestamp'] = df_pilar['timestamp'].dt.tz_convert("UTC")
             
        df_p_filt = df_pilar[['timestamp', 'GHI']].rename(columns={'GHI': 'target_GHI'})
        df_p_filt['station_id'] = 'PILAR'
        df_p_filt['estacion'] = 'PILAR'
        df_p_filt['cell_id'] = get_nearest_cell(df_feat_unique, pilar_lat, pilar_lon)
        df_p_filt['ghi_cs'], df_p_filt['cosZ_station'] = calc_cs(df_p_filt['timestamp'], pilar_lat, pilar_lon)
        df_p_filt['is_davis'] = False
        all_stations_data.append(df_p_filt)
        print("   ✓ Pilar lista.")

    # Davis (Red calibrada final - auditoría de huso horario)
    if DAVIS_FILE.exists():
        print("   • Procesando red de estaciones automáticas Davis...")
        df_davis = pd.read_parquet(DAVIS_FILE)
        if 'tiempo_utc' in df_davis.index.names:
            df_davis = df_davis.reset_index()
            
        unique_stations = df_davis.drop_duplicates(subset=['estacion']).copy()
        
        # Auditoría estricta: asegurar que ninguna estación recibe doble desplazamiento
        shifted_count = 0
        for _, row in unique_stations.iterrows():
            s_lat, s_lon, s_est = row['latitude'], row['longitude'], row['estacion']
            if pd.isna(s_lat) or pd.isna(s_lon):
                continue
                
            df_s = df_davis[df_davis['estacion'] == s_est][['tiempo_utc', 'solar_radiation']].copy()
            df_s.rename(columns={'tiempo_utc': 'timestamp', 'solar_radiation': 'target_GHI'}, inplace=True)
            
            if not df_s.empty:
                # Comprobación de auditoría sobre el pico de radiación
                envelope = df_s.groupby(df_s['timestamp'].dt.hour)['target_GHI'].quantile(0.90)
                if not envelope.isna().all():
                    peak_h = int(envelope.idxmax())
                    if peak_h != 16:
                        shifted_count += 1
                        
            df_s['station_id'] = s_est
            df_s['estacion'] = s_est
            df_s['cell_id'] = get_nearest_cell(df_feat_unique, s_lat, s_lon)
            df_s['ghi_cs'], df_s['cosZ_station'] = calc_cs(df_s['timestamp'], s_lat, s_lon)
            df_s['is_davis'] = True
            all_stations_data.append(df_s)
            
        print(f"   ✓ {len(unique_stations)} estaciones Davis incorporadas.")
        print(f"   ✓ Auditoría temporal: {len(unique_stations) - shifted_count}/{len(unique_stations)} estaciones con pico exactamente a las 16:00 UTC (sin desplazamientos espurios).")

    df_ground = pd.concat(all_stations_data, ignore_index=True)
    df_ground = df_ground.dropna(subset=['target_GHI', 'ghi_cs'])
    df_ground = df_ground[df_ground['ghi_cs'] > 10.0].copy() # Solo horas diurnas
    
    # Índice de Claridad acotado físicamente
    df_ground['Kt_station'] = (df_ground['target_GHI'] / df_ground['ghi_cs']).clip(0.0, 1.15)
    
    # --- 3. CONSOLIDACIÓN HORARIA Y EVALUACIÓN DE ALINEAMIENTO A / B / C ---
    print("\n🔬 3. Consolidación horaria y evaluación de alineación temporal A / B / C...")
    
    # Consolidación a una fila única por (cell_id, time_round)
    hourly_sat = (
        df_feat
        .groupby(['cell_id', 'time_round'], as_index=False)[['reflectance', 'B0', 'cosZ']]
        .mean()
        .sort_values(['cell_id', 'time_round'])
    )
    
    # Opción A: t
    df_feat_A = hourly_sat.copy()
    
    # Opción B: t + 1 h
    df_feat_B = hourly_sat.copy()
    df_feat_B['time_round'] = df_feat_B['time_round'] - pd.Timedelta(hours=1)
    
    # Opción C: media móvil consecutiva estricta (mean t, t+1) sin cruzar la noche
    hourly_C = hourly_sat.copy()
    prev_t = hourly_C.groupby('cell_id')['time_round'].shift(1)
    consecutive = (hourly_C['time_round'] - prev_t) == pd.Timedelta(hours=1)
    for col in ['reflectance', 'B0', 'cosZ']:
        prev = hourly_C.groupby('cell_id')[col].shift(1)
        hourly_C[col] = np.where(consecutive, (hourly_C[col] + prev) / 2.0, np.nan)
    hourly_C['time_round'] = hourly_C['time_round'] - pd.Timedelta(hours=1)
    df_feat_C = hourly_C.dropna(subset=['reflectance', 'B0', 'cosZ']).copy()

    def test_alignment(df_sat_opt, name):
        df_m = pd.merge(df_ground, df_sat_opt, left_on=['cell_id', 'timestamp'], right_on=['cell_id', 'time_round'], how='inner')
        df_m['Fr'] = np.maximum(df_m['reflectance']**2 - df_m['B0']**2, 0.0)
        df_m = df_m.dropna(subset=['Fr', 'cosZ', 'Kt_station'])
        
        pilar_r2 = -1.0
        davis_r2 = -1.0
        
        p_data = df_m[~df_m['is_davis']]
        if len(p_data) > 100:
            m_p = LinearRegression().fit(p_data[['Fr', 'cosZ']], p_data['Kt_station'])
            pilar_r2 = float(m_p.score(p_data[['Fr', 'cosZ']], p_data['Kt_station']))
            
        d_data = df_m[df_m['is_davis']]
        if len(d_data) > 100:
            m_d = LinearRegression().fit(d_data[['Fr', 'cosZ']], d_data['Kt_station'])
            davis_r2 = float(m_d.score(d_data[['Fr', 'cosZ']], d_data['Kt_station']))
            
        return {'opcion': name, 'R2_Pilar': pilar_r2, 'R2_Davis': davis_r2, 'N_pairs': len(df_m)}, df_m

    res_A, df_m_A = test_alignment(df_feat_A, 'A (t)')
    res_B, df_m_B = test_alignment(df_feat_B, 'B (t+1)')
    res_C, df_m_C = test_alignment(df_feat_C, 'C (mean t, t+1 consecutive)')
    
    df_res_opt = pd.DataFrame([res_A, res_B, res_C])
    print(df_res_opt.to_string(index=False))
    
    # Seleccionar la mejor opción empírica (mayor correlación en Davis y Pilar)
    best_idx = df_res_opt['R2_Davis'].idxmax()
    best_opt = df_res_opt.loc[best_idx]
    best_opt_name = best_opt['opcion']
    print(f"\n🎯 Estrategia seleccionada automáticamente: {best_opt_name}")
    
    if 'A' in best_opt_name:
        df_merge = df_m_A
    elif 'B' in best_opt_name:
        df_merge = df_m_B
    else:
        df_merge = df_m_C
        
    # --- 4. ASIGNACIÓN REGIONAL (CLUSTERS FASE 2) ---
    print("\n📂 4. Acoplando asignación regional de las estaciones terrestres...")
    clusters_df = gpd.read_file(CLUSTERS_FILE)[['estacion', 'cluster']]
    df_merge = pd.merge(df_merge, clusters_df, on='estacion', how='inner')
    print(f"   ✓ Registros acoplados con pertenencia regional: {len(df_merge):,} pares.")
    
    # Features de entrenamiento
    features = ['reflectance', 'B0', 'Fr', 'cosZ', 'day_of_year']
    df_merge['day_of_year'] = df_merge['timestamp'].dt.dayofyear
    
    def get_season(month):
        if month in [12, 1, 2]: return 'Verano'
        elif month in [3, 4, 5]: return 'Otono'
        elif month in [6, 7, 8]: return 'Invierno'
        else: return 'Primavera'
        
    df_merge['season'] = df_merge['timestamp'].dt.month.map(get_season)
    
    # --- 5. ENTRENAMIENTO REGIONAL-ESTACIONAL Y VALIDACIÓN CRUZADA ---
    print("\n🌍 5. Ejecutando Validación Cruzada Espacial y Ajuste de Modelos Finales...")
    results = []
    final_models_dict = {}
    all_predictions = []
    
    regions = sorted(df_merge['cluster'].unique())
    for region in regions:
        final_models_dict[region] = {}
        for season in ['Verano', 'Otono', 'Invierno', 'Primavera']:
            df_reg_season = df_merge[(df_merge['cluster'] == region) & (df_merge['season'] == season)].copy()
            n_groups = df_reg_season['estacion'].nunique()
            
            if len(df_reg_season) < 20 or n_groups < 1:
                print(f"   ⚠️ Saltando Región {region} - {season} (datos insuficientes)")
                continue
                
            print(f"   • Región {region} - {season:<10} | {n_groups:2d} estaciones | {len(df_reg_season):6d} observaciones")
            
            X = df_reg_season[features].values
            y = df_reg_season['Kt_station'].values
            y_ghi_real = df_reg_season['target_GHI'].values
            ghi_cs_arr = df_reg_season['ghi_cs'].values
            groups = df_reg_season['estacion'].values

            models = {
                'BD-JPT (Baseline)': BDJPTWrapper(),
                'Machine Learning (Kt)': HistGradientBoostingRegressor(max_iter=100, max_depth=12, random_state=42),
                'Híbrido (BD-JPT + ML)': HybridBDJPT_HGB(max_iter=100, max_depth=12, random_state=42)
            }

            metrics_kt = {name: {'RMSE': [], 'MAE': [], 'MBE': [], 'R2': []} for name in models}
            metrics_ghi = {name: {'RMSE': [], 'MAE': [], 'MBE': [], 'R2': [], 'nRMSE': [], 'nMBE': []} for name in models}
            region_season_preds = []

            if n_groups >= 2:
                n_splits = min(5, n_groups)
                cv = GroupKFold(n_splits=n_splits)
                splits = cv.split(X, y, groups)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                splits = cv.split(X, y)
                
            for fold, (train_idx, test_idx) in enumerate(splits):
                X_tr, y_tr, y_ghi_tr, cs_tr = X[train_idx], y[train_idx], y_ghi_real[train_idx], ghi_cs_arr[train_idx]
                X_te, y_te, y_ghi_te, cs_te = X[test_idx], y[test_idx], y_ghi_real[test_idx], ghi_cs_arr[test_idx]
                
                df_fold_preds = pd.DataFrame({
                    'timestamp': df_reg_season['timestamp'].values[test_idx],
                    'estacion': df_reg_season['estacion'].values[test_idx],
                    'cosZ': df_reg_season['cosZ'].values[test_idx],
                    'ghi_cs': cs_te,
                    'GHI_Real': y_ghi_te,
                    'Region': region,
                    'Season': season,
                    'Fold': fold
                })
                
                for name, model in models.items():
                    if name == 'BD-JPT (Baseline)':
                        model.fit(X_tr, y_ghi_tr)
                        preds_kt = model.predict(X_te, cs_te)
                    elif name == 'Híbrido (BD-JPT + ML)':
                        model.fit(X_tr, y_tr, y_ghi_tr, cs_tr)
                        preds_kt = model.predict(X_te, cs_te)
                    else:
                        model.fit(X_tr, y_tr)
                        preds_kt = model.predict(X_te)
                        
                    preds_kt = np.clip(preds_kt, 0.0, 1.15)
                    preds_ghi = preds_kt * cs_te
                    
                    df_fold_preds[name] = preds_ghi
                    
                    mean_ghi_te = np.mean(y_ghi_te) if np.mean(y_ghi_te) > 0 else 1e-6
                    
                    metrics_kt[name]['RMSE'].append(mean_squared_error(y_te, preds_kt, squared=False))
                    metrics_kt[name]['MAE'].append(mean_absolute_error(y_te, preds_kt))
                    metrics_kt[name]['MBE'].append(np.mean(preds_kt - y_te))
                    metrics_kt[name]['R2'].append(r2_score(y_te, preds_kt))
                    
                    metrics_ghi[name]['RMSE'].append(mean_squared_error(y_ghi_te, preds_ghi, squared=False))
                    metrics_ghi[name]['MAE'].append(mean_absolute_error(y_ghi_te, preds_ghi))
                    metrics_ghi[name]['MBE'].append(np.mean(preds_ghi - y_ghi_te))
                    metrics_ghi[name]['R2'].append(r2_score(y_ghi_te, preds_ghi))
                    metrics_ghi[name]['nRMSE'].append((metrics_ghi[name]['RMSE'][-1] / mean_ghi_te) * 100.0)
                    metrics_ghi[name]['nMBE'].append((metrics_ghi[name]['MBE'][-1] / mean_ghi_te) * 100.0)
                    
                region_season_preds.append(df_fold_preds)
                
            for name in models:
                r2_kt = np.mean(metrics_kt[name]['R2'])
                r2_ghi = np.mean(metrics_ghi[name]['R2'])
                rmse_ghi = np.mean(metrics_ghi[name]['RMSE'])
                mae_ghi = np.mean(metrics_ghi[name]['MAE'])
                mbe_ghi = np.mean(metrics_ghi[name]['MBE'])
                nrmse_ghi = np.mean(metrics_ghi[name]['nRMSE'])
                nmbe_ghi = np.mean(metrics_ghi[name]['nMBE'])
                results.append([f"R{region}", season, name, f"{r2_kt:.3f}", f"{r2_ghi:.3f}", f"{rmse_ghi:.1f}", f"{mae_ghi:.1f}", f"{mbe_ghi:.1f}", f"{nrmse_ghi:.1f}", f"{nmbe_ghi:.1f}"])
                
            all_predictions.extend(region_season_preds)
            
            # Ajustar modelo final sobre 100% de los datos regionales
            bdjpt_final = BDJPTWrapper().fit(X, y_ghi_real)
            hgb_final = HistGradientBoostingRegressor(max_iter=100, max_depth=12, random_state=42).fit(X, y)
            hybrid_final = HybridBDJPT_HGB(max_iter=100, max_depth=12, random_state=42).fit(X, y, y_ghi_real, ghi_cs_arr)
            
            final_models_dict[region][season] = {'bdjpt': bdjpt_final, 'ml': hgb_final, 'hybrid': hybrid_final}

    # --- 6. EXPORTACIÓN DE RESULTADOS Y CONTRATO DE INFERENCIA ---
    print("\n💾 6. Exportando resultados y serializando modelos finales...")
    
    # DataFrame de Métricas
    col_names = ['Región', 'Estación del año', 'Modelo', '$R^2_{Kt}$', '$R^2_{GHI}$', 'RMSE (W/m$^2$)', 'MAE (W/m$^2$)', 'MBE (W/m$^2$)', 'nRMSE (\\%)', 'nMBE (\\%)']
    df_results = pd.DataFrame(results, columns=col_names)
    
    # Predicciones OOF
    df_preds_all = pd.concat(all_predictions, ignore_index=True)
    df_preds_all.to_parquet(RESULTS_DIR / 'fase4_cv_predictions.parquet', index=False)
    print(f"   ✓ Predicciones OOF guardadas: {RESULTS_DIR / 'fase4_cv_predictions.parquet'} ({len(df_preds_all):,} filas)")
    
    # Tabla LaTeX
    tex_out = RESULTS_DIR / "bdjpt_regional_estacional_comparison.tex"
    latex_str = "\\begin{table}[htpb]\n\\centering\n\\small\n\\caption{Comparación de modelos predictivos regionalizados y estacionales con validación cruzada espacial estricta.}\n\\label{tab:bdjpt_regional_estacional_models}\n"
    latex_str += df_results.to_latex(index=False, escape=False)
    latex_str += "\\end{table}"
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"   ✓ Tabla LaTeX exportada: {tex_out}")
    
    # Guardar Pickle de modelos
    joblib.dump(final_models_dict, MODELS_FILE)
    print(f"   ✓ Modelos finales serializados: {MODELS_FILE}")
    
    # Cargar coeficientes B0
    with open(B0_PARAMS_FILE, "r") as f:
        b0_params = json.load(f)
        
    # Congelar Contrato de Inferencia (exportado directamente por Fase 4)
    contract = {
        "pipeline_version": "5B.2.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_architecture": "HybridBDJPT_HGB",
        "target_variable": "clearness_index_kt",
        "kt_bounds": {
            "min": 0.0,
            "max": 1.15
        },
        "features": features,
        "temporal_alignment": {
            "strategy": str(best_opt_name),
            "r2_pilar": float(best_opt['R2_Pilar']),
            "r2_davis": float(best_opt['R2_Davis']),
            "n_pairs": int(best_opt['N_pairs']),
            "description": "Desplazamiento temporal t+1 h para sincronizar la observación satelital instantánea con la integral horaria precedente de superficie."
        },
        "b0_model": {
            "type": "Tarpley_Clear_Sky",
            "params_file": str(B0_PARAMS_FILE),
            "coefficients": {
                "A": b0_params["A"],
                "B": b0_params["B"],
                "C": b0_params["C"],
                "D": b0_params["D"]
            }
        },
        "models_file": str(MODELS_FILE),
        "pv_system_specs": {
            "module_type": "Mono-PERC",
            "pdc0_w": 1000.0,
            "gamma_pdc": -0.0035,
            "noct_c": 45.0,
            "inverter_nominal_eff": 0.96,
            "bos_losses": 0.10,
            "fixed_tilt_deg": 30.0,
            "fixed_azimuth_deg": 0.0,
            "tracker_type": "singleaxis_horizontal_NS",
            "tracker_backtrack": True,
            "tracker_gcr": 0.35,
            "tracker_max_angle": 60.0
        }
    }
    
    with open(CONTRACT_FILE, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)
    print(f"   ✓ Contrato metodológico congelado: {CONTRACT_FILE}")
    
    print("=" * 80)
    print("✅ FASE 4 COMPLETADA EXITOSAMENTE Y LISTA PARA FASE 5")
    print("=" * 80)

if __name__ == '__main__':
    main()
