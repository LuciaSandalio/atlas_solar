#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 5B: Inferencia y Modelado PV Exclusivo de 2025 + QA de Transición Satelital
==============================================================================
Convenio ATLAS II — FCEFyN / UNC & OHMC

1. Ingesta los 365 días de reflectancia de 2025 (data/processed_2km/reflectancia_2025*.parquet).
2. Aplica la cadena física y de inferencia canónica:
   - Ineichen Clearsky con turbidez Linke mensual y altimetría topográfica.
   - Tarpley B0 y contraste nuboso Fr.
   - Inferencia de Kt con HybridBDJPT_HGB (Opción C: mean(t, t+1)).
   - Descomposición espectro-angular Erbs (DNI, DHI).
   - Transposición anisotrópica Perez: Fijo 30°N y Tracker 1-eje N-S con backtracking (GCR=0.35, max 60°).
   - Modelo térmico NOCT (45°C) y modelo paramétrico PVWatts (DC + Inversor AC 96% + 10% BOS).
3. Genera los productos anuales y mensuales de 2025:
   - data/ready/atlas_definitivo_2km/atlas_anual_2025.parquet
   - data/ready/atlas_definitivo_2km/atlas_monthly_2025.npy
   - data/ready/atlas_definitivo_2km/qa_diario_2025.parquet
4. Ejecuta el protocolo de QA científico sobre la transición GOES-16 -> GOES-19 (abril 2025).
"""

import sys
import os
import time
import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import pvlib
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from models.bdjpt import BDJPTWrapper, HybridBDJPT_HGB

DATA_DIR = ROOT_DIR / "data"
PROCESSED_2KM_DIR = DATA_DIR / "processed_2km"
READY_DIR = DATA_DIR / "ready"
DEF_DIR = READY_DIR / "atlas_definitivo_2km"
DEF_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR = ROOT_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Variables globales para workers multiproceso
g_alt = None
g_pres = None
g_linke = None
g_mod_C = None
g_n_cells = None
g_in_prov = None


def init_worker():
    global g_alt, g_pres, g_linke, g_mod_C, g_n_cells, g_in_prov
    g_alt = np.load(READY_DIR / "altitude_grid_2km.npy")
    g_pres = np.load(READY_DIR / "pressure_grid_2km.npy")
    g_linke = np.load(READY_DIR / "linke_turbidity_grid_2km.npy")
    g_mod_C = joblib.load(READY_DIR / "b0_model" / "model_hybrid_bdjpt_hgb_optionC.pkl")
    g_n_cells = len(g_alt)
    grid = pd.read_parquet(READY_DIR / "cordoba_grid_2km_regions.parquet")
    g_in_prov = grid['in_province'].values.astype(bool)


def process_single_day(fpath):
    global g_alt, g_pres, g_linke, g_mod_C, g_n_cells, g_in_prov
    
    fname = Path(fpath).name
    day_str = fname.replace("reflectancia_", "").replace(".parquet", "")
    
    try:
        df_day = pd.read_parquet(fpath)
    except Exception as e:
        print(f"Error leyendo {fname}: {e}")
        return None
        
    if df_day.empty:
        return None
        
    df_day['hour'] = df_day['timestamp'].dt.hour
    agg = df_day.groupby(['cell_id', 'hour'])[['reflectance', 'zenith_deg', 'azimuth_deg']].mean().reset_index()
    hours = sorted(agg['hour'].unique())
    if len(hours) == 0:
        return None
        
    # Detectar satélite predominante del archivo
    sat_name = df_day['satellite'].iloc[0] if 'satellite' in df_day.columns else "unknown"
        
    hour_dict = {}
    for h in hours:
        sub = agg[agg['hour'] == h].sort_values('cell_id')
        hour_dict[h] = {
            'ref': sub['reflectance'].values.astype(np.float64),
            'zen': sub['zenith_deg'].values.astype(np.float64),
            'az': sub['azimuth_deg'].values.astype(np.float64)
        }
        
    doy = int(pd.to_datetime(day_str).dayofyear)
    month_idx = int(pd.to_datetime(day_str).month) - 1
    
    # Acumuladores diarios espaciales (59,837 celdas)
    daily_ghi = np.zeros(g_n_cells, dtype=np.float32)
    daily_dni = np.zeros(g_n_cells, dtype=np.float32)
    daily_dhi = np.zeros(g_n_cells, dtype=np.float32)
    daily_poa_fix = np.zeros(g_n_cells, dtype=np.float32)
    daily_poa_trk = np.zeros(g_n_cells, dtype=np.float32)
    daily_yf_fix = np.zeros(g_n_cells, dtype=np.float32)
    daily_yf_trk = np.zeros(g_n_cells, dtype=np.float32)
    
    # Métricas para QA diario provincial
    sum_ref_prov = 0.0
    sum_b0_prov = 0.0
    sum_fr_prov = 0.0
    sum_kt_prov = 0.0
    sum_ghics_prov = 0.0
    count_obs_hours = 0
    
    A, B, C, D = 0.2289, -0.0156, -0.0096, 0.0853
    pdc0, gamma_pdc, noct, t_ref, eta_inv_nom, eta_bos = 1000.0, -0.0035, 45.0, 25.0, 0.96, 0.90
    season_month = month_idx + 1
    t_amb = 24.0 if season_month in [12, 1, 2] else 18.0 if season_month in [3, 4, 5] else 11.0 if season_month in [6, 7, 8] else 19.0
    dni_extra = pvlib.irradiance.get_extra_radiation(doy)
    lt = g_linke[:, month_idx]
    
    for i, h in enumerate(hours):
        h_data = hour_dict[h]
        # Estrategia C: promedio temporal continuo con la hora siguiente si es consecutiva
        if i < len(hours) - 1 and hours[i+1] == h + 1:
            ref_c = 0.5 * (h_data['ref'] + hour_dict[h+1]['ref'])
        else:
            ref_c = h_data['ref']
            
        zen_true = h_data['zen']
        az = h_data['az']
        
        # Filtro de noche astronómica
        if np.all(zen_true >= 85.0):
            continue
            
        # Cenit aparente con refracción SPA exacta
        elev = 90.0 - zen_true
        press_mbar = g_pres / 100.0
        delta_e = (press_mbar / 1010.0) * (283.0 / 285.0) * 1.02 / (
            60.0 * np.tan(np.radians(np.maximum(elev, 0.1) + 10.3 / (np.maximum(elev, 0.1) + 5.11)))
        )
        app_zen = np.clip(zen_true - delta_e, 0.0, 90.0)
        
        # Clearsky Ineichen
        am_rel = pvlib.atmosphere.get_relative_airmass(app_zen)
        am_abs = pvlib.atmosphere.get_absolute_airmass(am_rel, g_pres)
        cs = pvlib.clearsky.ineichen(app_zen, am_abs, lt, altitude=g_alt, dni_extra=dni_extra)
        ghi_cs = np.maximum(np.asarray(cs['ghi']), 0.0)
        
        # Tarpley B0 y Contraste Fr
        z_rad = np.radians(zen_true)
        cosZ = np.cos(z_rad)
        sinZ = np.sin(z_rad)
        cosGamma = np.cos(np.radians(az))
        b0 = np.clip(A + B*cosZ + C*sinZ*cosGamma + D*sinZ*(cosGamma**2), 0.01, None)
        fr = np.maximum(ref_c**2 - b0**2, 0.0)
        
        # Inferencia de Kt con HybridBDJPT_HGB (Opción C)
        X = np.column_stack([ref_c, b0, fr, cosZ, np.full(g_n_cells, doy, dtype=np.float64)])
        kt = g_mod_C.predict(X, ghi_cs)
        kt = np.clip(kt, 0.0, 1.15)
        ghi = kt * ghi_cs
        
        # Acumular estadísticas diurnas en provincia para QA
        mask_sun = (ghi_cs[g_in_prov] > 50.0)
        if np.any(mask_sun):
            sum_ref_prov += ref_c[g_in_prov][mask_sun].mean()
            sum_b0_prov += b0[g_in_prov][mask_sun].mean()
            sum_fr_prov += fr[g_in_prov][mask_sun].mean()
            sum_kt_prov += kt[g_in_prov][mask_sun].mean()
            sum_ghics_prov += ghi_cs[g_in_prov][mask_sun].mean()
            count_obs_hours += 1
        
        # Descomposición Erbs
        t_stamp = pd.Timestamp(f'{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]} {h:02d}:00:00+00:00')
        erbs_res = pvlib.irradiance.erbs(ghi, zen_true, pd.DatetimeIndex([t_stamp] * g_n_cells))
        dni = np.nan_to_num(np.maximum(np.asarray(erbs_res['dni']), 0.0), nan=0.0)
        dhi = np.nan_to_num(np.maximum(np.asarray(erbs_res['dhi']), 0.0), nan=0.0)
        
        # Transposición Fijo 30° Norte
        poa_fix = pvlib.irradiance.get_total_irradiance(
            surface_tilt=30.0, surface_azimuth=0.0,
            solar_zenith=app_zen, solar_azimuth=az,
            dni=dni, ghi=ghi, dhi=dhi, dni_extra=dni_extra, model='perez'
        )
        g_poa_fix = np.nan_to_num(np.maximum(np.asarray(poa_fix['poa_global']), 0.0), nan=0.0)
        
        # Transposición Tracker 1 eje N-S con backtracking (GCR 0.35)
        trk = pvlib.tracking.singleaxis(
            apparent_zenith=app_zen, solar_azimuth=az,
            axis_tilt=0.0, axis_azimuth=0.0,
            max_angle=60.0, backtrack=True, gcr=0.35
        )
        s_tilt = np.nan_to_num(np.asarray(trk['surface_tilt']), nan=0.0)
        s_az = np.nan_to_num(np.asarray(trk['surface_azimuth']), nan=0.0)
        
        poa_trk = pvlib.irradiance.get_total_irradiance(
            surface_tilt=s_tilt, surface_azimuth=s_az,
            solar_zenith=app_zen, solar_azimuth=az,
            dni=dni, ghi=ghi, dhi=dhi, dni_extra=dni_extra, model='perez'
        )
        g_poa_trk = np.nan_to_num(np.maximum(np.asarray(poa_trk['poa_global']), 0.0), nan=0.0)
        
        # Modelo Térmico NOCT y Generación PVWatts (kW/kWp)
        t_cell_fix = t_amb + ((noct - 20.0) / 800.0) * g_poa_fix
        t_cell_trk = t_amb + ((noct - 20.0) / 800.0) * g_poa_trk
        pdc_fix = pvlib.pvsystem.pvwatts_dc(g_poa_fix, t_cell_fix, pdc0, gamma_pdc, t_ref)
        pac_fix = (np.maximum(pvlib.inverter.pvwatts(pdc_fix, pdc0, eta_inv_nom), 0.0) * eta_bos) / 1000.0
        pdc_trk = pvlib.pvsystem.pvwatts_dc(g_poa_trk, t_cell_trk, pdc0, gamma_pdc, t_ref)
        pac_trk = (np.maximum(pvlib.inverter.pvwatts(pdc_trk, pdc0, eta_inv_nom), 0.0) * eta_bos) / 1000.0
        
        # Acumular hora (Delta t = 1.0 h)
        daily_ghi += ghi.astype(np.float32)
        daily_dni += dni.astype(np.float32)
        daily_dhi += dhi.astype(np.float32)
        daily_poa_fix += g_poa_fix.astype(np.float32)
        daily_poa_trk += g_poa_trk.astype(np.float32)
        daily_yf_fix += pac_fix.astype(np.float32)
        daily_yf_trk += pac_trk.astype(np.float32)
        
    qa_record = {
        'day_str': day_str,
        'doy': doy,
        'satellite': sat_name,
        'mean_ref': float(sum_ref_prov / count_obs_hours) if count_obs_hours > 0 else np.nan,
        'mean_b0': float(sum_b0_prov / count_obs_hours) if count_obs_hours > 0 else np.nan,
        'mean_fr': float(sum_fr_prov / count_obs_hours) if count_obs_hours > 0 else np.nan,
        'mean_kt': float(sum_kt_prov / count_obs_hours) if count_obs_hours > 0 else np.nan,
        'mean_ghi_cs_kwh': float((sum_ghics_prov / count_obs_hours) * count_obs_hours / 1000.0) if count_obs_hours > 0 else np.nan,
        'ghi_kwh_prov': float(daily_ghi[g_in_prov].mean() / 1000.0),
        'dni_kwh_prov': float(daily_dni[g_in_prov].mean() / 1000.0),
        'yf_fix_prov': float(daily_yf_fix[g_in_prov].mean()),
        'yf_trk_prov': float(daily_yf_trk[g_in_prov].mean())
    }
        
    return (day_str, month_idx, daily_ghi, daily_dni, daily_dhi, daily_poa_fix, daily_poa_trk, daily_yf_fix, daily_yf_trk, qa_record)


def main():
    print("=" * 80)
    print("🚀 FASE 5B: INFERENCIA Y MODELADO PV DEL AÑO 2025 (365 DÍAS COMPLETOS)")
    print("=" * 80)
    
    # 1. Grilla provincial
    grid = pd.read_parquet(READY_DIR / "cordoba_grid_2km_regions.parquet")
    n_cells = len(grid)
    lats = grid['lat'].values
    lons = grid['lon'].values
    in_prov = grid['in_province'].values.astype(bool)
    print(f"📂 Grilla provincial 2 km cargada: {n_cells:,} celdas ({in_prov.sum():,} dentro de provincia).")
    
    flist_2025 = sorted(glob.glob(str(PROCESSED_2KM_DIR / "reflectancia_2025*.parquet")))
    print(f"📂 Archivos satelitales de 2025 encontrados: {len(flist_2025)} días.")
    if len(flist_2025) != 365:
        print(f"⚠️ Advertencia: se esperaban 365 días, pero se encontraron {len(flist_2025)}.")
        
    num_workers = min(8, cpu_count())
    print(f"\n⚡ Inferencia masiva y modelado PV para 2025 con {num_workers} procesos paralelos...")
    t_start = time.time()
    
    y_ghi = np.zeros(n_cells, dtype=np.float32)
    y_dni = np.zeros(n_cells, dtype=np.float32)
    y_dhi = np.zeros(n_cells, dtype=np.float32)
    y_poa_fix = np.zeros(n_cells, dtype=np.float32)
    y_poa_trk = np.zeros(n_cells, dtype=np.float32)
    y_yf_fix = np.zeros(n_cells, dtype=np.float32)
    y_yf_trk = np.zeros(n_cells, dtype=np.float32)
    monthly_2025 = np.zeros((n_cells, 12), dtype=np.float32)
    qa_records = []
    
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        res_iter = pool.imap_unordered(process_single_day, flist_2025, chunksize=4)
        
        for idx, r in enumerate(res_iter):
            if r is None:
                continue
            day_str, m_idx, d_ghi, d_dni, d_dhi, d_p_fix, d_p_trk, d_y_fix, d_y_trk, qa_rec = r
            
            kwh_ghi = d_ghi / 1000.0
            kwh_dni = d_dni / 1000.0
            kwh_dhi = d_dhi / 1000.0
            kwh_poa_fix = d_p_fix / 1000.0
            kwh_poa_trk = d_p_trk / 1000.0
            
            y_ghi += kwh_ghi
            y_dni += kwh_dni
            y_dhi += kwh_dhi
            y_poa_fix += kwh_poa_fix
            y_poa_trk += kwh_poa_trk
            y_yf_fix += d_y_fix
            y_yf_trk += d_y_trk
            
            monthly_2025[:, m_idx] += kwh_ghi
            qa_records.append(qa_rec)
            
            if (idx + 1) % 50 == 0 or (idx + 1) == len(flist_2025):
                print(f"   [2025] Procesados {idx+1}/{len(flist_2025)} días ({(time.time()-t_start):.1f} s)...")
                
    elapsed = time.time() - t_start
    print(f"\n✅ Procesamiento 2025 completado en {elapsed/60:.2f} minutos.")
    
    # 2. Guardar producto anual 2025
    ann_2025_file = DEF_DIR / "atlas_anual_2025.parquet"
    monthly_2025_file = DEF_DIR / "atlas_monthly_2025.npy"
    
    df_2025 = pd.DataFrame({
        'cell_id': grid['cell_id'], 'lat': lats, 'lon': lons, 'in_province': in_prov,
        'ghi_kwh_m2': y_ghi, 'dni_kwh_m2': y_dni, 'dhi_kwh_m2': y_dhi,
        'poa_fixed_kwh_m2': y_poa_fix, 'poa_tracker_kwh_m2': y_poa_trk,
        'yf_fixed_kwh_kwp': y_yf_fix, 'yf_tracker_kwh_kwp': y_yf_trk
    })
    df_2025.to_parquet(ann_2025_file, index=False)
    np.save(monthly_2025_file, monthly_2025)
    print(f"\n💾 Producto anual 2025 guardado:")
    print(f"   • Parquet anual : {ann_2025_file} ({ann_2025_file.stat().st_size / 1e6:.2f} MB)")
    print(f"   • Matriz mensual: {monthly_2025_file}")
    print(f"   • GHI medio provincial 2025 : {y_ghi[in_prov].mean():.1f} kWh/m²/año")
    print(f"   • DNI medio provincial 2025 : {y_dni[in_prov].mean():.1f} kWh/m²/año")
    print(f"   • Yf Fijo medio 2025        : {y_yf_fix[in_prov].mean():.1f} kWh/kWp/año")
    print(f"   • Yf Tracker medio 2025     : {y_yf_trk[in_prov].mean():.1f} kWh/kWp/año")
    
    # 3. Guardar serie QA diaria
    df_qa = pd.DataFrame(qa_records).sort_values('day_str').reset_index(drop=True)
    qa_file = DEF_DIR / "qa_diario_2025.parquet"
    df_qa.to_parquet(qa_file, index=False)
    print(f"   • Registro QA diario        : {qa_file} ({len(df_qa)} días registrados)")
    
    # --------------------------------------------------------------------------
    # 4. PROTOCOLO DE QA DE TRANSICIÓN SATELITAL (GOES-16 -> GOES-19)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔍 AUDITORÍA DE TRANSICIÓN SATELITAL: GOES-16 -> GOES-19 (ABRIL 2025)")
    print("=" * 80)
    
    # Ventana de transición: 15 días antes (días 83 a 97) vs 15 días después (días 98 a 112)
    # Traspaso oficial: 8 de abril de 2025 = DOY 98
    doy_trans = 98
    df_pre = df_qa[(df_qa['doy'] >= doy_trans - 15) & (df_qa['doy'] < doy_trans)]
    df_post = df_qa[(df_qa['doy'] >= doy_trans) & (df_qa['doy'] < doy_trans + 15)]
    
    print(f"Ventana Pre-Transición (DOY {doy_trans-15} a {doy_trans-1}, N={len(df_pre)} días, GOES-16):")
    print(f"   • Reflectancia media diurna : {df_pre['mean_ref'].mean():.4f} ± {df_pre['mean_ref'].std():.4f}")
    print(f"   • Contraste nuboso Fr medio : {df_pre['mean_fr'].mean():.4f} ± {df_pre['mean_fr'].std():.4f}")
    print(f"   • Índice de claridad Kt     : {df_pre['mean_kt'].mean():.4f} ± {df_pre['mean_kt'].std():.4f}")
    print(f"   • GHI diaria provincial     : {df_pre['ghi_kwh_prov'].mean():.2f} ± {df_pre['ghi_kwh_prov'].std():.2f} kWh/m²/día")
    
    print(f"\nVentana Post-Transición (DOY {doy_trans} a {doy_trans+14}, N={len(df_post)} días, GOES-19):")
    print(f"   • Reflectancia media diurna : {df_post['mean_ref'].mean():.4f} ± {df_post['mean_ref'].std():.4f}")
    print(f"   • Contraste nuboso Fr medio : {df_post['mean_fr'].mean():.4f} ± {df_post['mean_fr'].std():.4f}")
    print(f"   • Índice de claridad Kt     : {df_post['mean_kt'].mean():.4f} ± {df_post['mean_kt'].std():.4f}")
    print(f"   • GHI diaria provincial     : {df_post['ghi_kwh_prov'].mean():.2f} ± {df_post['ghi_kwh_prov'].std():.2f} kWh/m²/día")
    
    # Comparación en días despejados (Kt > 0.65)
    pre_clear = df_pre[df_pre['mean_kt'] >= 0.65]
    post_clear = df_post[df_post['mean_kt'] >= 0.65]
    print(f"\nDías de Cielo Claro en la Transición:")
    print(f"   • Pre (GOES-16, N={len(pre_clear)}) : Kt={pre_clear['mean_kt'].mean():.4f}, Ref={pre_clear['mean_ref'].mean():.4f}, Fr={pre_clear['mean_fr'].mean():.4f}")
    print(f"   • Post (GOES-19, N={len(post_clear)}) : Kt={post_clear['mean_kt'].mean():.4f}, Ref={post_clear['mean_ref'].mean():.4f}, Fr={post_clear['mean_fr'].mean():.4f}")
    
    diff_kt_clear = post_clear['mean_kt'].mean() - pre_clear['mean_kt'].mean() if (len(pre_clear) > 0 and len(post_clear) > 0) else 0.0
    diff_ref_clear = post_clear['mean_ref'].mean() - pre_clear['mean_ref'].mean() if (len(pre_clear) > 0 and len(post_clear) > 0) else 0.0
    print(f"\n📊 Diagnóstico de Continuidad:")
    print(f"   • Delta Kt en cielo claro   : {diff_kt_clear:+.4f} ({diff_kt_clear*100:+.2f}%)")
    print(f"   • Delta Ref en cielo claro  : {diff_ref_clear:+.4f}")
    if abs(diff_kt_clear) < 0.03:
        print("   ✅ CONCLUSIÓN QA: Transición suave y continua. No se detecta escalón artificial sistemático.")
    else:
        print("   ⚠️ ATENCIÓN: Desviación observable entre períodos.")
    print("=" * 80)


if __name__ == "__main__":
    main()
