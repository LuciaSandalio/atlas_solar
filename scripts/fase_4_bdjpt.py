import pandas as pd
import numpy as np
import pvlib
import joblib
import geopandas as gpd
from pathlib import Path
from scipy.optimize import lsq_linear
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_DIR = Path('data')
PROCESSED_DIR = DATA_DIR / 'processed'
FEATURES_FILE = Path('data/ready/b0_model/features_with_B0.parquet')
PILAR_FILE = PROCESSED_DIR / 'stations' / 'pilar_cmp21_hourly.parquet'
DAVIS_FILE = PROCESSED_DIR / 'stations' / 'davis_calibrated_hourly_final.parquet'
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

class BDJPTWrapper:
    def __init__(self, Isc=1361.0):
        self.Isc = Isc
        self.a, self.b, self.c = 0.0, 0.0, 0.0
        self.d_clear, self.d_cloudy = 0.0, 0.0

    def fit(self, X, target_ghi):
        # X features: [reflectance, B0, Fr, cosZ, day_of_year]
        ref, B0, Fr, cosZ = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        
        # Step 1: Clear sky fit (a,b,c)
        thresh_dynamic = B0 + 0.05
        mask_clear = ref <= thresh_dynamic
        mask_cloudy = ref > thresh_dynamic
        
        # Step 1: Clear sky fit (a,b,c)
        mask_strictly_clear = ref <= (B0 + 0.02)
        if np.sum(mask_strictly_clear) < 500:
            mask_strictly_clear = ref <= np.quantile(ref, 0.15)
            
        X_strict = np.column_stack([
            self.Isc * cosZ[mask_strictly_clear],
            self.Isc * (cosZ[mask_strictly_clear]**2),
            self.Isc * (cosZ[mask_strictly_clear]**3)
        ])
        
        if len(X_strict) > 0:
            res_cs = lsq_linear(X_strict, target_ghi[mask_strictly_clear], bounds=(-3.0, 3.0))
            self.a, self.b, self.c = res_cs.x
        else:
            self.a, self.b, self.c = 1.0, 0.0, 0.0
            
        # Step 2: Cloud index fit (d1, d2)
        GHI_cs_calc = self.Isc * cosZ * (self.a + self.b*cosZ + self.c*(cosZ**2))
        
        def fit_d(mask):
            if np.sum(mask) == 0: return 0.0
            X4 = Fr[mask].reshape(-1, 1)
            y_diff = target_ghi[mask] - GHI_cs_calc[mask]
            res = lsq_linear(X4, y_diff, bounds=(-10000.0, 0.0))
            return res.x[0]
            
        self.d_clear = fit_d(mask_clear)
        self.d_cloudy = fit_d(mask_cloudy)
        return self

    def predict(self, X, ghi_cs_ineichen):
        ref, B0, Fr, cosZ = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        GHI_cs_calc = self.Isc * cosZ * (self.a + self.b*cosZ + self.c*(cosZ**2))
        
        pred_ghi = np.zeros(len(X))
        
        thresh_dynamic = B0 + 0.05
        mask_clear = ref <= thresh_dynamic
        mask_cloudy = ref > thresh_dynamic
        
        pred_ghi[mask_clear] = GHI_cs_calc[mask_clear] + self.d_clear * Fr[mask_clear]
        pred_ghi[mask_cloudy] = GHI_cs_calc[mask_cloudy] + self.d_cloudy * Fr[mask_cloudy]
        
        pred_ghi = np.maximum(pred_ghi, 0.0)
        # Transform physical GHI prediction back to Kt domain for fair comparison
        return pred_ghi / ghi_cs_ineichen

class HybridBDJPT_RF:
    """Modelo híbrido: BD-JPT como base física + RF para el residual"""
    def __init__(self, lambda_shrinkage=1.0):
        self.bdjpt = BDJPTWrapper()
        self.rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
        self.lambda_shrinkage = lambda_shrinkage
        self.residual_mean = 0.0
        
    def fit(self, X, y_kt, y_ghi, cs_tr):
        self.bdjpt.fit(X, y_ghi)
        preds_kt_bdjpt = self.bdjpt.predict(X, cs_tr)
        
        residual = y_kt - preds_kt_bdjpt
        self.residual_mean = np.mean(residual)
        residual_centered = residual - self.residual_mean
        
        X_hybrid = np.column_stack([X, preds_kt_bdjpt])
        self.rf.fit(X_hybrid, residual_centered)
        return self
        
    def predict(self, X, cs_te):
        preds_kt_bdjpt = self.bdjpt.predict(X, cs_te)
        
        X_hybrid = np.column_stack([X, preds_kt_bdjpt])
        preds_residual = self.rf.predict(X_hybrid) + self.residual_mean
        
        kt_hybrid = preds_kt_bdjpt + (self.lambda_shrinkage * preds_residual)
        return np.clip(kt_hybrid, 0.0, 1.15)

def calc_cs(timestamps, lat, lon):
    times = pd.DatetimeIndex(timestamps)
    loc = pvlib.location.Location(lat, lon)
    cs = loc.get_clearsky(times)
    solpos = loc.get_solarposition(times)
    return cs['ghi'].values, np.maximum(np.cos(np.radians(solpos['apparent_zenith'])), 0.0)

def main():
    print("🚀 Fase 4: Entrenando Modelos Satelitales (Kt)")
    
    # --- 1. CARGAR FEATURES SATELITALES ---
    print("Cargando features satelitales...")
    df_feat = pd.read_parquet(FEATURES_FILE)
    df_feat['time_round'] = pd.to_datetime(df_feat['time_round'], utc=True)
    df_feat_unique = df_feat[['cell_id', 'lat', 'lon']].drop_duplicates()
    
    def get_nearest_cell(df_u, lat, lon):
        dist = (df_u['lat'] - lat)**2 + (df_u['lon'] - lon)**2
        return df_u.loc[dist.idxmin(), 'cell_id']

    # --- 2. PREPARAR DATOS TERRESTRES ---
    all_stations_data = []
    
    # Pilar
    pilar_lat, pilar_lon = -31.667, -63.883
    if PILAR_FILE.exists():
        print("Procesando estación Pilar...")
        df_pilar = pd.read_parquet(PILAR_FILE)
        if df_pilar['timestamp'].dt.tz is None:
             df_pilar['timestamp'] = df_pilar['timestamp'].dt.tz_localize("UTC")
        else:
             df_pilar['timestamp'] = df_pilar['timestamp'].dt.tz_convert("UTC")
             
        df_p_filt = df_pilar[['timestamp', 'GHI']].rename(columns={'GHI': 'target_GHI'})
        df_p_filt['station_id'] = 'PILAR'
        df_p_filt['cell_id'] = get_nearest_cell(df_feat_unique, pilar_lat, pilar_lon)
        df_p_filt['ghi_cs'], df_p_filt['cosZ_station'] = calc_cs(df_p_filt['timestamp'], pilar_lat, pilar_lon)
        df_p_filt['is_davis'] = False
        all_stations_data.append(df_p_filt)
        print("✅ Pilar lista.")

    # Davis
    if DAVIS_FILE.exists():
        print("Procesando estaciones Davis...")
        df_davis = pd.read_parquet(DAVIS_FILE)
        if 'tiempo_utc' in df_davis.index.names: df_davis.reset_index(inplace=True)
        
        stations = df_davis.drop_duplicates(subset=['station_id', 'latitude', 'longitude']).copy()
        
        for _, row in stations.iterrows():
            s_lat, s_lon, s_id = row['latitude'], row['longitude'], row['station_id']
            if pd.isna(s_lat) or pd.isna(s_lon): continue
            
            df_s = df_davis[df_davis['station_id'] == s_id][['tiempo_utc', 'solar_radiation']].copy()
            df_s.rename(columns={'tiempo_utc': 'timestamp', 'solar_radiation': 'target_GHI'}, inplace=True)
            
            if not df_s.empty:
                envelope = df_s.groupby(df_s['timestamp'].dt.hour)['target_GHI'].quantile(0.90)
                if not envelope.isna().all():
                    peak_hour = int(envelope.idxmax())
                    shift_hours = 16 - peak_hour
                    if shift_hours != 0:
                        df_s['timestamp'] = df_s['timestamp'] + pd.Timedelta(hours=shift_hours)
            
            df_s['station_id'] = s_id
            df_s['cell_id'] = get_nearest_cell(df_feat_unique, s_lat, s_lon)
            df_s['ghi_cs'], df_s['cosZ_station'] = calc_cs(df_s['timestamp'], s_lat, s_lon)
            df_s['is_davis'] = True
            all_stations_data.append(df_s)
        print(f"✅ {len(stations)} estaciones Davis mapeadas y corregidas temporalmente.")

    df_ground = pd.concat(all_stations_data, ignore_index=True)
    df_ground = df_ground.dropna(subset=['target_GHI', 'ghi_cs'])
    df_ground = df_ground[df_ground['ghi_cs'] > 10.0].copy() # Solo diurno
    
    # Calcular Kt (Clampeado)
    df_ground['Kt_station'] = (df_ground['target_GHI'] / df_ground['ghi_cs']).clip(0, 1.15)
    
    # --- 3. EVALUACIÓN A/B/C ---
    print("\nEvaluando opciones de alineación temporal A/B/C...")
    df_feat_sorted = df_feat.sort_values(['cell_id', 'time_round'])
    
    # Option A: t
    df_feat_A = df_feat_sorted.copy()
    
    # Option B: t+1
    df_feat_B = df_feat_sorted.copy()
    df_feat_B['time_round'] = df_feat_B['time_round'] - pd.Timedelta(hours=1)
    
    # Option C: mean(t, t+1)
    df_feat_C = df_feat_sorted.copy()
    df_feat_C[['reflectance', 'B0', 'cosZ']] = df_feat_C.groupby('cell_id')[['reflectance', 'B0', 'cosZ']].rolling(2).mean().reset_index(0, drop=True)
    df_feat_C['time_round'] = df_feat_C['time_round'] - pd.Timedelta(hours=1) # The end interval matches t in surface

    def test_alignment(df_sat_opt, name):
        df_m = pd.merge(df_ground, df_sat_opt, left_on=['cell_id', 'timestamp'], right_on=['cell_id', 'time_round'], how='inner')
        df_m['Fr'] = np.maximum(df_m['reflectance']**2 - df_m['B0']**2, 0)
        df_m = df_m.dropna(subset=['Fr', 'cosZ', 'Kt_station'])
        
        # Test separately for Pilar and Davis
        pilar_r2 = -1
        davis_r2 = -1
        
        if len(df_m[~df_m['is_davis']]) > 100:
            p_data = df_m[~df_m['is_davis']]
            model = LinearRegression().fit(p_data[['Fr', 'cosZ']], p_data['Kt_station'])
            pilar_r2 = model.score(p_data[['Fr', 'cosZ']], p_data['Kt_station'])
            
        if len(df_m[df_m['is_davis']]) > 100:
            d_data = df_m[df_m['is_davis']]
            model = LinearRegression().fit(d_data[['Fr', 'cosZ']], d_data['Kt_station'])
            davis_r2 = model.score(d_data[['Fr', 'cosZ']], d_data['Kt_station'])
            
        return {'opcion': name, 'R2_Pilar': pilar_r2, 'R2_Davis': davis_r2}, df_m

    res_A, df_m_A = test_alignment(df_feat_A, 'A (t)')
    res_B, df_m_B = test_alignment(df_feat_B, 'B (t+1)')
    res_C, df_m_C = test_alignment(df_feat_C, 'C (mean t,t+1)')
    
    df_res_opt = pd.DataFrame([res_A, res_B, res_C])
    print(df_res_opt)
    
    best_opt = df_res_opt.loc[df_res_opt['R2_Davis'].idxmax()]
    print(f"\nSeleccionando Opción {best_opt['opcion']} para el entrenamiento final...")
    
    if best_opt['opcion'] == 'A (t)': df_merge = df_m_A
    elif best_opt['opcion'] == 'B (t+1)': df_merge = df_m_B
    else: df_merge = df_m_C

    # Load Clusters and Filter
    print("\nCargando asignación regional de estaciones (Fase 2)...")
    clusters_df = gpd.read_file('data/processed/stations/davis_solar_clusters.gpkg')[['estacion', 'cluster']]
    clusters_df['station_id'] = clusters_df['estacion'].str.split('_').str[0].astype(float)
    clusters_df = clusters_df[['station_id', 'cluster']]
    
    # Cast df_merge station_id to float (Pilar becomes NaN, which is fine since it will drop in inner join)
    df_merge['station_id'] = pd.to_numeric(df_merge['station_id'], errors='coerce')
    
    # Merge on 'station_id'. Pilar will be dropped, and only the 166 approved Davis stations will be kept.
    df_merge = pd.merge(df_merge, clusters_df, on='station_id', how='inner')
    
    print("\n--- Pipeline Regional Multi-Modelo (Kt) ---")
    features = ['reflectance', 'B0', 'Fr', 'cosZ']
    df_merge['day_of_year'] = df_merge['timestamp'].dt.dayofyear
    features += ['day_of_year']
    
    def get_season(month):
        if month in [12, 1, 2]: return 'Verano'
        elif month in [3, 4, 5]: return 'Otono'
        elif month in [6, 7, 8]: return 'Invierno'
        else: return 'Primavera'
        
    df_merge['season'] = df_merge['timestamp'].dt.month.map(get_season)
    
    results = []
    final_models_dict = {}
    
    for region in sorted(df_merge['cluster'].unique()):
        final_models_dict[region] = {}
        for season in ['Verano', 'Otono', 'Invierno', 'Primavera']:
            
            df_reg_season = df_merge[(df_merge['cluster'] == region) & (df_merge['season'] == season)].copy()
            
            n_groups = df_reg_season['station_id'].nunique()
            if len(df_reg_season) < 100 or n_groups < 2:
                print(f"Saltando Región {region} - {season} (datos insuficientes o n_groups < 2)")
                continue
                
            print(f"\n🌍 Entrenando Región {region} - 🌤️ {season} (n={n_groups} est, {len(df_reg_season)} obs)...")
            
            X = df_reg_season[features].values
            y = df_reg_season['Kt_station'].values
            y_ghi_real = df_reg_season['target_GHI'].values
            ghi_cs_arr = df_reg_season['ghi_cs'].values
            groups = df_reg_season['station_id'].astype(str).values

            models = {
                'BD-JPT (Baseline)': BDJPTWrapper(),
                'RandomForest (Kt)': RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
                'Híbrido (BD-JPT + RF)': HybridBDJPT_RF()
            }

            metrics_kt = {name: {'RMSE': [], 'MAE': [], 'MBE': [], 'R2': []} for name in models}
            metrics_ghi = {name: {'RMSE': [], 'MAE': [], 'MBE': [], 'R2': []} for name in models}

            n_splits = min(5, n_groups)
            gkf = GroupKFold(n_splits=n_splits)
            for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
                X_tr, y_tr, y_ghi_tr, cs_tr = X[train_idx], y[train_idx], y_ghi_real[train_idx], ghi_cs_arr[train_idx]
                X_te, y_te, y_ghi_te, cs_te = X[test_idx], y[test_idx], y_ghi_real[test_idx], ghi_cs_arr[test_idx]
                
                for name, model in models.items():
                    if name == 'BD-JPT (Baseline)':
                        model.fit(X_tr, y_ghi_tr)
                        preds_kt = model.predict(X_te, cs_te)
                    elif name == 'Híbrido (BD-JPT + RF)':
                        model.fit(X_tr, y_tr, y_ghi_tr, cs_tr)
                        preds_kt = model.predict(X_te, cs_te)
                    else:
                        model.fit(X_tr, y_tr)
                        preds_kt = model.predict(X_te)
                        
                    preds_kt = np.clip(preds_kt, 0, 1.15)
                    preds_ghi = preds_kt * cs_te
                    
                    metrics_kt[name]['RMSE'].append(mean_squared_error(y_te, preds_kt, squared=False))
                    metrics_kt[name]['MAE'].append(mean_absolute_error(y_te, preds_kt))
                    metrics_kt[name]['MBE'].append(np.mean(preds_kt - y_te))
                    metrics_kt[name]['R2'].append(r2_score(y_te, preds_kt))
                    
                    metrics_ghi[name]['RMSE'].append(mean_squared_error(y_ghi_te, preds_ghi, squared=False))
                    metrics_ghi[name]['MAE'].append(mean_absolute_error(y_ghi_te, preds_ghi))
                    metrics_ghi[name]['MBE'].append(np.mean(preds_ghi - y_ghi_te))
                    metrics_ghi[name]['R2'].append(r2_score(y_ghi_te, preds_ghi))
                    
            print(f"--- Resultados Región {region} - {season} ---")
            for name in models:
                r2_kt = np.mean(metrics_kt[name]['R2'])
                r2_ghi = np.mean(metrics_ghi[name]['R2'])
                rmse_ghi = np.mean(metrics_ghi[name]['RMSE'])
                mae_ghi = np.mean(metrics_ghi[name]['MAE'])
                mbe_ghi = np.mean(metrics_ghi[name]['MBE'])
                print(f"{name:<20} | R2_Kt: {r2_kt:.3f} | R2_GHI: {r2_ghi:.3f} | RMSE_GHI: {rmse_ghi:.1f} W/m2")
                results.append([f"R{region}", season, name, f"{r2_kt:.3f}", f"{r2_ghi:.3f}", f"{rmse_ghi:.1f}", f"{mae_ghi:.1f}", f"{mbe_ghi:.1f}"])

            # Train final regional models on 100% of the regional data
            bdjpt_final = BDJPTWrapper()
            bdjpt_final.fit(X, y_ghi_real)
            
            rf_final = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
            rf_final.fit(X, y)
            
            hybrid_final = HybridBDJPT_RF()
            hybrid_final.fit(X, y, y_ghi_real, ghi_cs_arr)
            
            final_models_dict[region][season] = {'bdjpt': bdjpt_final, 'ml': rf_final, 'hybrid': hybrid_final}

    # Consolidated export
    print("\n--- Resultados Consolidados ---")
    df_results = pd.DataFrame(results, columns=['Región', 'Estación', 'Modelo', '$R^2_{Kt}$', '$R^2_{GHI}$', 'RMSE (W/m$^2$)', 'MAE (W/m$^2$)', 'MBE (W/m$^2$)'])
    
    tex_out = RESULTS_DIR / "bdjpt_regional_estacional_comparison.tex"
    latex_str = "\\begin{table}[htpb]\n\\centering\n\\caption{Comparación de modelos predictivos regionalizados y estacionales con validación cruzada espacial estricta.}\n\\label{tab:bdjpt_regional_estacional_models}\n"
    latex_str += df_results.to_latex(index=False, escape=False)
    latex_str += "\\end{table}"
    with open(tex_out, "w") as f:
        f.write(latex_str)
    
    joblib.dump(final_models_dict, PROCESSED_DIR / 'bdjpt_final_models_regional_estacional.pkl')
    print("\nTodos los modelos regionales-estacionales guardados exitosamente.")

if __name__ == '__main__':
    main()
