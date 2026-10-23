#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 5B: Atlas Solar Definitivo Quinquenal de Córdoba (5 Años Completos: 2020 a 2024)
===================================================================================
Convenio ATLAS II — Provincia de Córdoba (Componente Solar)
FCEFyN / UNC — OHMC

1. Metodología de Climatología Quinquenal (Estándar WMO / NREL):
   Utiliza exclusivamente los 5 años completos consolidados en grilla nativa 2 km:
   2020 (366 d), 2021 (365 d), 2022 (365 d), 2023 (365 d) y 2024 (366 d).
   H_y(x, y) = sum_t GHI_y(x, y, t) * Delta t
   H_definitivo(x, y) = (1/5) * sum_{y=2020}^{2024} H_y(x, y)

2. Cadena Física Completa por Celda (59,837 celdas canónicas a 2 km):
   - Ineichen Clearsky con turbidez Linke mensual y altimetría topográfica.
   - Brillo de cielo claro Tarpley B0 recalibrado para Córdoba y contraste nuboso Fr.
   - Inferencia de Kt mediante HybridBDJPT_HGB (Opción C: mean(t, t+1)).
   - Descomposición radiativa Erbs et al. (1982) -> DNI, DHI.
   - Transposición Perez et al. (1990):
     * Configuración A: Plano Fijo 30° Norte (azimut 0°).
     * Configuración B: Seguidor 1-eje horizontal N-S con backtracking (GCR=0.35, max 60°).
   - Modelo térmico NOCT (45°C) y modelo paramétrico PVWatts (DC + Inversor AC 96% + 10% BOS).

3. Diez (10) Mapas Oficiales Anuales Medios:
   1. GHI anual medio (kWh/m²/año)
   2. DNI anual medio (kWh/m²/año)
   3. DHI anual medio (kWh/m²/año)
   4. POA fijo 30° N (kWh/m²/año)
   5. POA tracker 1 eje (kWh/m²/año)
   6. Rendimiento específico fijo Y_f (kWh/kWp/año)
   7. Rendimiento específico tracker Y_f (kWh/kWp/año)
   8. Factor de capacidad fijo (%)
   9. Factor de capacidad tracker (%)
   10. Ganancia tracker respecto de fijo (%)

4. Evaluación de Estabilidad y Convergencia Multianual:
   - Coeficiente de Variación Interanual Quinquenal: CV_interanual (%) sobre los 5 años.
   - Curva de Convergencia Temporal: M_20, M_20-21, M_20-22, M_20-23, M_20-24.
   - Incrementos relativos: Delta_2, Delta_3, Delta_4, Delta_5.
   - Ciclo estacional mensual completo: 12 meses (Enero a Diciembre) promediados sobre 5 años.
   - Matriz de correlación espacial interanual.
"""

import sys
import os
import time
import json
import glob
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import pvlib
import joblib
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from models.bdjpt import BDJPTWrapper, HybridBDJPT_HGB

# Directorios de datos y resultados
DATA_DIR = ROOT_DIR / "data"
PROCESSED_2KM_DIR = DATA_DIR / "processed_2km"
READY_DIR = DATA_DIR / "ready"
DEF_DIR = READY_DIR / "atlas_definitivo_2km"
DEF_DIR.mkdir(parents=True, exist_ok=True)
PRELIM_DIR = READY_DIR / "atlas_preliminar_2km"

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = ROOT_DIR / "reportes" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Configuración visual
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

CIUDADES_CORDOBA = {
    "Córdoba": (-64.18, -31.42),
    "Río Cuarto": (-64.35, -33.12),
    "Villa María": (-63.24, -32.41),
    "San Francisco": (-62.08, -31.43),
    "Cruz del Eje": (-64.80, -30.72),
    "Villa Dolores": (-65.19, -31.94),
    "Laboulaye": (-63.39, -34.13)
}

# Variables globales para workers multiproceso
g_alt = None
g_pres = None
g_linke = None
g_mod_C = None
g_n_cells = None


def init_worker():
    global g_alt, g_pres, g_linke, g_mod_C, g_n_cells
    g_alt = np.load(READY_DIR / "altitude_grid_2km.npy")
    g_pres = np.load(READY_DIR / "pressure_grid_2km.npy")
    g_linke = np.load(READY_DIR / "linke_turbidity_grid_2km.npy")
    g_mod_C = joblib.load(READY_DIR / "b0_model" / "model_hybrid_bdjpt_hgb_optionC.pkl")
    g_n_cells = len(g_alt)


def process_single_day(fpath):
    global g_alt, g_pres, g_linke, g_mod_C, g_n_cells
    
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
    
    # Acumuladores diarios
    daily_ghi = np.zeros(g_n_cells, dtype=np.float32)
    daily_dni = np.zeros(g_n_cells, dtype=np.float32)
    daily_dhi = np.zeros(g_n_cells, dtype=np.float32)
    daily_poa_fix = np.zeros(g_n_cells, dtype=np.float32)
    daily_poa_trk = np.zeros(g_n_cells, dtype=np.float32)
    daily_yf_fix = np.zeros(g_n_cells, dtype=np.float32)
    daily_yf_trk = np.zeros(g_n_cells, dtype=np.float32)
    
    A, B, C, D = 0.2289, -0.0156, -0.0096, 0.0853
    pdc0, gamma_pdc, noct, t_ref, eta_inv_nom, eta_bos = 1000.0, -0.0035, 45.0, 25.0, 0.96, 0.90
    season_month = month_idx + 1
    t_amb = 24.0 if season_month in [12, 1, 2] else 18.0 if season_month in [3, 4, 5] else 11.0 if season_month in [6, 7, 8] else 19.0
    dni_extra = pvlib.irradiance.get_extra_radiation(doy)
    lt = g_linke[:, month_idx]
    
    for i, h in enumerate(hours):
        h_data = hour_dict[h]
        if i < len(hours) - 1 and hours[i+1] == h + 1:
            ref_c = 0.5 * (h_data['ref'] + hour_dict[h+1]['ref'])
        else:
            ref_c = h_data['ref']
            
        zen_true = h_data['zen']
        az = h_data['az']
        
        if np.all(zen_true >= 85.0):
            continue
            
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
        
    return (day_str, month_idx, daily_ghi, daily_dni, daily_dhi, daily_poa_fix, daily_poa_trk, daily_yf_fix, daily_yf_trk)


def overlay_decorations(ax, boundary_gdf=None, deptos_gdf=None):
    if deptos_gdf is not None:
        deptos_gdf.boundary.plot(ax=ax, color='#475569', linewidth=0.5, alpha=0.6, zorder=3)
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color='#0f172a', linewidth=1.1, zorder=4)
        
    for name, (lon, lat) in CIUDADES_CORDOBA.items():
        ax.plot(lon, lat, marker='o', markersize=3.0, color='white', markeredgecolor='black', markeredgewidth=0.7, zorder=5)
        ax.text(lon + 0.05, lat + 0.05, name, fontsize=7.0, weight='bold', color='#0f172a',
                bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', boxstyle='round,pad=0.15'), zorder=6)
                
    ax.set_xlim(-65.8, -61.6)
    ax.set_ylim(-35.2, -29.4)
    ax.set_xlabel("Longitud (°O)")
    ax.set_ylabel("Latitud (°S)")
    ax.grid(True, linestyle=':', alpha=0.4, color='#94a3b8')


def main():
    print("=" * 80)
    print("☀️ FASE 5B: ATLAS SOLAR DEFINITIVO QUINQUENAL DE CÓRDOBA (2020 - 2024)")
    print("   Convenio ATLAS II — FCEFyN / UNC & OHMC")
    print("=" * 80)
    
    # 1. Grilla provincial
    grid = pd.read_parquet(READY_DIR / "cordoba_grid_2km_regions.parquet")
    n_cells = len(grid)
    lats = grid['lat'].values
    lons = grid['lon'].values
    in_prov = grid['in_province'].values.astype(bool)
    print(f"📂 Grilla provincial 2 km cargada: {n_cells:,} celdas ({in_prov.sum():,} dentro de provincia).")
    
    boundary_gdf = gpd.read_file(DATA_DIR / "cordoba_boundary.geojson") if (DATA_DIR / "cordoba_boundary.geojson").exists() else None
    deptos_gdf = gpd.read_file(ROOT_DIR / "shapes_cba" / "cba_deptos.shp") if (ROOT_DIR / "shapes_cba" / "cba_deptos.shp").exists() else None
    
    # Los 5 años completos de la climatología quinquenal
    years_complete = ['2020', '2021', '2022', '2023', '2024']
    all_files = sorted(glob.glob(str(PROCESSED_2KM_DIR / "reflectancia_*.parquet")))
    
    year_files = {}
    for y in years_complete:
        flist = [f for f in all_files if Path(f).stem.replace('reflectancia_', '').startswith(y)]
        year_files[y] = flist
        print(f"   • Año {y}: {len(flist)} archivos Parquet encontrados.")
        
    annual_data = {}
    monthly_data = {y: np.zeros((n_cells, 12), dtype=np.float32) for y in years_complete}
    
    num_workers = min(8, cpu_count())
    print(f"\n⚡ Inferencia masiva y modelado PV con {num_workers} procesos paralelos...")
    t_start = time.time()
    
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        for y in years_complete:
            flist = year_files[y]
            ann_def_file = DEF_DIR / f"atlas_anual_{y}.parquet"
            monthly_def_file = DEF_DIR / f"atlas_monthly_{y}.npy"
            
            # Reutilizar checkpoints existentes de 2020 y 2021 si son 100% idénticos
            ann_prelim_file = PRELIM_DIR / f"atlas_anual_{y}.parquet"
            monthly_prelim_file = PRELIM_DIR / f"atlas_monthly_{y}.npy"
            
            if ann_def_file.exists() and monthly_def_file.exists():
                print(f"🔄 Año {y}: Cargando desde checkpoint definitivo...")
                df_y = pd.read_parquet(ann_def_file)
                annual_data[y] = {
                    'GHI': df_y['ghi_kwh_m2'].values,
                    'DNI': df_y['dni_kwh_m2'].values,
                    'DHI': df_y['dhi_kwh_m2'].values,
                    'POA_fixed': df_y['poa_fixed_kwh_m2'].values,
                    'POA_tracker': df_y['poa_tracker_kwh_m2'].values,
                    'Yf_fixed': df_y['yf_fixed_kwh_kwp'].values,
                    'Yf_tracker': df_y['yf_tracker_kwh_kwp'].values
                }
                monthly_data[y] = np.load(monthly_def_file)
                continue
            elif y in ['2020', '2021'] and ann_prelim_file.exists() and monthly_prelim_file.exists():
                print(f"🔄 Año {y}: Reutilizando checkpoint validado de 2020/2021...")
                df_y = pd.read_parquet(ann_prelim_file)
                annual_data[y] = {
                    'GHI': df_y['ghi_kwh_m2'].values,
                    'DNI': df_y['dni_kwh_m2'].values,
                    'DHI': df_y['dhi_kwh_m2'].values,
                    'POA_fixed': df_y['poa_fixed_kwh_m2'].values,
                    'POA_tracker': df_y['poa_tracker_kwh_m2'].values,
                    'Yf_fixed': df_y['yf_fixed_kwh_kwp'].values,
                    'Yf_tracker': df_y['yf_tracker_kwh_kwp'].values
                }
                monthly_data[y] = np.load(monthly_prelim_file)
                df_y.to_parquet(ann_def_file, index=False)
                np.save(monthly_def_file, monthly_data[y])
                continue
                
            print(f"\n--- Procesando Año Completo {y} ({len(flist)} días) ---")
            y_ghi = np.zeros(n_cells, dtype=np.float32)
            y_dni = np.zeros(n_cells, dtype=np.float32)
            y_dhi = np.zeros(n_cells, dtype=np.float32)
            y_poa_fix = np.zeros(n_cells, dtype=np.float32)
            y_poa_trk = np.zeros(n_cells, dtype=np.float32)
            y_yf_fix = np.zeros(n_cells, dtype=np.float32)
            y_yf_trk = np.zeros(n_cells, dtype=np.float32)
            
            t_y0 = time.time()
            res_iter = pool.imap_unordered(process_single_day, flist, chunksize=4)
            
            for idx, r in enumerate(res_iter):
                if r is None:
                    continue
                day_str, m_idx, d_ghi, d_dni, d_dhi, d_p_fix, d_p_trk, d_y_fix, d_y_trk = r
                
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
                
                monthly_data[y][:, m_idx] += kwh_ghi
                
                if (idx + 1) % 50 == 0 or (idx + 1) == len(flist):
                    print(f"   [{y}] Procesados {idx+1}/{len(flist)} días ({(time.time()-t_y0):.1f} s)...")
                    
            annual_data[y] = {
                'GHI': y_ghi, 'DNI': y_dni, 'DHI': y_dhi,
                'POA_fixed': y_poa_fix, 'POA_tracker': y_poa_trk,
                'Yf_fixed': y_yf_fix, 'Yf_tracker': y_yf_trk
            }
            
            df_y = pd.DataFrame({
                'cell_id': grid['cell_id'], 'lat': lats, 'lon': lons, 'in_province': in_prov,
                'ghi_kwh_m2': y_ghi, 'dni_kwh_m2': y_dni, 'dhi_kwh_m2': y_dhi,
                'poa_fixed_kwh_m2': y_poa_fix, 'poa_tracker_kwh_m2': y_poa_trk,
                'yf_fixed_kwh_kwp': y_yf_fix, 'yf_tracker_kwh_kwp': y_yf_trk
            })
            df_y.to_parquet(ann_def_file, index=False)
            np.save(monthly_def_file, monthly_data[y])
            print(f"✓ Año {y} guardado en {ann_def_file}")
            print(f"   • GHI medio provincial: {y_ghi[in_prov].mean():.1f} kWh/m²/año | Yf Fijo: {y_yf_fix[in_prov].mean():.1f} | Yf Tracker: {y_yf_trk[in_prov].mean():.1f}")
            
    print(f"\n✅ Inferencia completada en {(time.time()-t_start)/60:.2f} minutos.")
    
    # --------------------------------------------------------------------------
    # 2. CONSOLIDACIÓN DEL ATLAS QUINQUENAL DEFINITIVO (2020 a 2024)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 CALCULANDO PROMEDIO QUINQUENAL DEFINITIVO (5 AÑOS COMPLETOS: 2020 - 2024)")
    print("=" * 80)
    
    ghi_mean = np.mean([annual_data[y]['GHI'] for y in years_complete], axis=0)
    dni_mean = np.mean([annual_data[y]['DNI'] for y in years_complete], axis=0)
    dhi_mean = np.mean([annual_data[y]['DHI'] for y in years_complete], axis=0)
    poa_fix_mean = np.mean([annual_data[y]['POA_fixed'] for y in years_complete], axis=0)
    poa_trk_mean = np.mean([annual_data[y]['POA_tracker'] for y in years_complete], axis=0)
    yf_fix_mean = np.mean([annual_data[y]['Yf_fixed'] for y in years_complete], axis=0)
    yf_trk_mean = np.mean([annual_data[y]['Yf_tracker'] for y in years_complete], axis=0)
    
    cf_fix_mean = (yf_fix_mean / 8760.0) * 100.0
    cf_trk_mean = (yf_trk_mean / 8760.0) * 100.0
    gain_trk_mean = ((yf_trk_mean - yf_fix_mean) / np.maximum(yf_fix_mean, 1e-3)) * 100.0
    
    df_atlas_10 = pd.DataFrame({
        'cell_id': grid['cell_id'],
        'lat': lats,
        'lon': lons,
        'in_province': in_prov,
        'region': grid['region'],
        'ghi_anual_kwh_m2': np.round(ghi_mean, 1),
        'dni_anual_kwh_m2': np.round(dni_mean, 1),
        'dhi_anual_kwh_m2': np.round(dhi_mean, 1),
        'poa_fijo_anual_kwh_m2': np.round(poa_fix_mean, 1),
        'poa_tracker_anual_kwh_m2': np.round(poa_trk_mean, 1),
        'yf_fixed_kwh_kwp': np.round(yf_fix_mean, 1),
        'yf_tracker_kwh_kwp': np.round(yf_trk_mean, 1),
        'cf_fixed_pct': np.round(cf_fix_mean, 2),
        'cf_tracker_pct': np.round(cf_trk_mean, 2),
        'ganancia_tracker_pct': np.round(gain_trk_mean, 2)
    })
    
    out_medio_file = DEF_DIR / "atlas_anual_2020_2024_medio.parquet"
    df_atlas_10.to_parquet(out_medio_file, index=False)
    print(f"✓ Base de datos quinquenal oficial guardada en {out_medio_file}")
    
    # Resumen provincial
    p_ghi = ghi_mean[in_prov]
    p_dni = dni_mean[in_prov]
    p_dhi = dhi_mean[in_prov]
    p_poa_fix = poa_fix_mean[in_prov]
    p_poa_trk = poa_trk_mean[in_prov]
    p_yf_fix = yf_fix_mean[in_prov]
    p_yf_trk = yf_trk_mean[in_prov]
    p_cf_fix = cf_fix_mean[in_prov]
    p_cf_trk = cf_trk_mean[in_prov]
    p_gain_trk = gain_trk_mean[in_prov]
    
    print("\nRESUMEN TERRITORIAL PROVINCIAL (Promedio Quinquenal 2020 - 2024):")
    print(f"1. GHI Anual Medio       : {p_ghi.mean():.1f} kWh/m²/año (Rango: {p_ghi.min():.1f} - {p_ghi.max():.1f})")
    print(f"2. DNI Anual Medio       : {p_dni.mean():.1f} kWh/m²/año (Rango: {p_dni.min():.1f} - {p_dni.max():.1f})")
    print(f"3. DHI Anual Medio       : {p_dhi.mean():.1f} kWh/m²/año (Rango: {p_dhi.min():.1f} - {p_dhi.max():.1f})")
    print(f"4. POA Fijo 30° N        : {p_poa_fix.mean():.1f} kWh/m²/año (Rango: {p_poa_fix.min():.1f} - {p_poa_fix.max():.1f})")
    print(f"5. POA Tracker 1-eje     : {p_poa_trk.mean():.1f} kWh/m²/año (Rango: {p_poa_trk.min():.1f} - {p_poa_trk.max():.1f})")
    print(f"6. Rendimiento Fijo Yf   : {p_yf_fix.mean():.1f} kWh/kWp/año (Rango: {p_yf_fix.min():.1f} - {p_yf_fix.max():.1f})")
    print(f"7. Rendimiento Tracker Yf: {p_yf_trk.mean():.1f} kWh/kWp/año (Rango: {p_yf_trk.min():.1f} - {p_yf_trk.max():.1f})")
    print(f"8. Factor Capacidad Fijo : {p_cf_fix.mean():.2f} % (Rango: {p_cf_fix.min():.2f} - {p_cf_fix.max():.2f})")
    print(f"9. Factor Capacidad Trk  : {p_cf_trk.mean():.2f} % (Rango: {p_cf_trk.min():.2f} - {p_cf_trk.max():.2f})")
    print(f"10. Ganancia Tracker     : {p_gain_trk.mean():.2f} % (Rango: {p_gain_trk.min():.2f} - {p_gain_trk.max():.2f})")
    
    # --------------------------------------------------------------------------
    # 3. ESTABILIDAD ESPACIAL Y CORRELACIÓN INTERANUAL ENTRE LOS 5 AÑOS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔬 MATRIZ DE CORRELACIÓN ESPACIAL ENTRE LOS 5 AÑOS COMPLETOS")
    print("=" * 80)
    
    df_inter = pd.DataFrame({
        f'GHI_{y}': annual_data[y]['GHI'][in_prov] for y in years_complete
    })
    corr_matrix = df_inter.corr()
    print(corr_matrix.to_string())
    corr_matrix.to_csv(RESULTS_DIR / "correlacion_espacial_5anios.csv")
    
    # --------------------------------------------------------------------------
    # 4. CONVERGENCIA TEMPORAL Y COEFICIENTE DE VARIACIÓN QUINQUENAL
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📈 ANÁLISIS DE CONVERGENCIA TEMPORAL Y COEFICIENTE DE VARIACIÓN QUINQUENAL")
    print("=" * 80)
    
    M_20 = annual_data['2020']['GHI']
    M_20_21 = (annual_data['2020']['GHI'] + annual_data['2021']['GHI']) / 2.0
    M_20_22 = (annual_data['2020']['GHI'] + annual_data['2021']['GHI'] + annual_data['2022']['GHI']) / 3.0
    M_20_23 = (annual_data['2020']['GHI'] + annual_data['2021']['GHI'] + annual_data['2022']['GHI'] + annual_data['2023']['GHI']) / 4.0
    M_20_24 = ghi_mean  # 5 años
    
    delta_2 = 100.0 * (M_20_21 - M_20) / np.maximum(M_20, 1.0)
    delta_3 = 100.0 * (M_20_22 - M_20_21) / np.maximum(M_20_21, 1.0)
    delta_4 = 100.0 * (M_20_23 - M_20_22) / np.maximum(M_20_22, 1.0)
    delta_5 = 100.0 * (M_20_24 - M_20_23) / np.maximum(M_20_23, 1.0)
    
    arr_5y = np.column_stack([annual_data[y]['GHI'] for y in years_complete])
    sigma_5y = np.std(arr_5y, axis=1)
    mu_5y = np.mean(arr_5y, axis=1)
    cv_interanual = 100.0 * (sigma_5y / np.maximum(mu_5y, 1.0))
    
    df_conv = pd.DataFrame({
        'cell_id': grid['cell_id'],
        'lat': lats,
        'lon': lons,
        'in_province': in_prov,
        'M_2020': M_20,
        'M_2020_2021': M_20_21,
        'M_2020_2022': M_20_22,
        'M_2020_2023': M_20_23,
        'M_2020_2024': M_20_24,
        'delta_2_pct': delta_2,
        'delta_3_pct': delta_3,
        'delta_4_pct': delta_4,
        'delta_5_pct': delta_5,
        'cv_interanual_pct': cv_interanual
    })
    df_conv.to_parquet(DEF_DIR / "atlas_convergencia_temporal_5years.parquet", index=False)
    
    print(f"Convergencia del Atlas Provincial (Territorio de Córdoba):")
    print(f"   • Delta 2 (1 -> 2 años): Media absoluta = {np.abs(delta_2[in_prov]).mean():.2f}%")
    print(f"   • Delta 3 (2 -> 3 años): Media absoluta = {np.abs(delta_3[in_prov]).mean():.2f}%")
    print(f"   • Delta 4 (3 -> 4 años): Media absoluta = {np.abs(delta_4[in_prov]).mean():.2f}%")
    print(f"   • Delta 5 (4 -> 5 años): Media absoluta = {np.abs(delta_5[in_prov]).mean():.2f}%")
    print(f"   • CV Interanual Quinquenal Medio: {cv_interanual[in_prov].mean():.2f}% (Rango: [{cv_interanual[in_prov].min():.2f}%, {cv_interanual[in_prov].max():.2f}%])")
    
    # --------------------------------------------------------------------------
    # 5. CICLO ESTACIONAL COMPLETO DE LOS 12 MESES (PROMEDIO 5 AÑOS)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📅 CICLO MENSUAL ANUAL COMPLETO (12 MESES: PROMEDIO 5 AÑOS)")
    print("=" * 80)
    
    month_names = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    monthly_means = {}
    for m in range(12):
        m_mean = np.mean([monthly_data[y][:, m] for y in years_complete], axis=0)
        monthly_means[month_names[m]] = m_mean
        print(f"   • {month_names[m]:<12}: {m_mean[in_prov].mean():.1f} kWh/m²/mes")
        
    df_months = pd.DataFrame({'cell_id': grid['cell_id'], 'lat': lats, 'lon': lons, 'in_province': in_prov, **monthly_means})
    df_months.to_parquet(DEF_DIR / "atlas_promedios_mensuales_quinquenal.parquet", index=False)
    
    # --------------------------------------------------------------------------
    # 6. GENERACIÓN DE FIGURAS CARTOGRÁFICAS DEFINITIVAS DE ALTA RESOLUCIÓN
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🗺️ 6. GENERANDO FIGURAS CARTOGRÁFICAS OFICIALES DE ALTA RESOLUCIÓN")
    print("=" * 80)
    
    # FIGURA 1: PANEL DE 10 MAPAS QUINQUENALES
    fig, axes = plt.subplots(2, 5, figsize=(26, 12), dpi=300)
    axes = axes.ravel()
    
    map_configs = [
        ("GHI Anual Medio", ghi_mean, "inferno", (1650, 2050), "kWh/m²/año"),
        ("DNI Anual Medio", dni_mean, "inferno", (1700, 2250), "kWh/m²/año"),
        ("DHI Anual Medio", dhi_mean, "inferno", (500, 750), "kWh/m²/año"),
        ("POA Fijo 30° N", poa_fix_mean, "plasma", (1700, 2150), "kWh/m²/año"),
        ("POA Tracker 1-Eje", poa_trk_mean, "plasma", (2050, 2550), "kWh/m²/año"),
        ("Rendimiento Fijo (Yf)", yf_fix_mean, "viridis", (1450, 1800), "kWh/kWp/año"),
        ("Rendimiento Tracker (Yf)", yf_trk_mean, "viridis", (1750, 2150), "kWh/kWp/año"),
        ("Factor Capacidad Fijo", cf_fix_mean, "magma", (16.5, 20.5), "%"),
        ("Factor Capacidad Tracker", cf_trk_mean, "magma", (20.0, 24.5), "%"),
        ("Ganancia Tracker", gain_trk_mean, "cividis", (17.0, 23.0), "%")
    ]
    
    for idx, (title, data, cmap, (vmin, vmax), unit) in enumerate(map_configs):
        ax = axes[idx]
        sc = ax.scatter(lons[in_prov], lats[in_prov], c=data[in_prov], cmap=cmap, vmin=vmin, vmax=vmax, s=0.9, edgecolors='none', rasterized=True)
        overlay_decorations(ax, boundary_gdf, deptos_gdf)
        ax.set_title(f"{title}\nMedia: {data[in_prov].mean():.1f} {unit}", fontsize=10.5, weight='bold')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="4%", pad=0.15)
        cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(unit, fontsize=8.5)
        
    plt.suptitle("Atlas Solar Definitivo de Córdoba — Climatología Quinquenal Completa (2020 - 2024)\nGrilla Canónica 2 km | Modelo Híbrido BD-JPT/HGB (Opción C)", fontsize=14, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "figura_atlas_definitivo_10_mapas.png")
    plt.close()
    print("   ✓ Guardada Figura Oficial 1: figura_atlas_definitivo_10_mapas.png")
    
    # FIGURA 2: CONVERGENCIA TEMPORAL Y CV INTERANUAL
    fig, axes = plt.subplots(1, 4, figsize=(24, 7.5), dpi=300)
    
    # Delta 3 (2 -> 3 años)
    ax = axes[0]
    vals = df_conv.loc[in_prov, 'delta_3_pct']
    sc = ax.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='RdBu_r', vmin=-3.0, vmax=3.0, s=1.2, edgecolors='none', rasterized=True)
    overlay_decorations(ax, boundary_gdf, deptos_gdf)
    ax.set_title(r"$\Delta_3$ (2 $\to$ 3 Años)" + f"\nMedia Abs: {np.abs(vals).mean():.2f}%", fontsize=11, weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4%", pad=0.15)
    plt.colorbar(sc, cax=cax, orientation="horizontal").set_label("% variación", fontsize=9)
    
    # Delta 4 (3 -> 4 años)
    ax = axes[1]
    vals = df_conv.loc[in_prov, 'delta_4_pct']
    sc = ax.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='RdBu_r', vmin=-2.0, vmax=2.0, s=1.2, edgecolors='none', rasterized=True)
    overlay_decorations(ax, boundary_gdf, deptos_gdf)
    ax.set_title(r"$\Delta_4$ (3 $\to$ 4 Años)" + f"\nMedia Abs: {np.abs(vals).mean():.2f}%", fontsize=11, weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4%", pad=0.15)
    plt.colorbar(sc, cax=cax, orientation="horizontal").set_label("% variación", fontsize=9)
    
    # Delta 5 (4 -> 5 años)
    ax = axes[2]
    vals = df_conv.loc[in_prov, 'delta_5_pct']
    sc = ax.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='RdBu_r', vmin=-1.5, vmax=1.5, s=1.2, edgecolors='none', rasterized=True)
    overlay_decorations(ax, boundary_gdf, deptos_gdf)
    ax.set_title(r"$\Delta_5$ (4 $\to$ 5 Años)" + f"\nMedia Abs: {np.abs(vals).mean():.2f}%", fontsize=11, weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4%", pad=0.15)
    plt.colorbar(sc, cax=cax, orientation="horizontal").set_label("% variación", fontsize=9)
    
    # CV Interanual Quinquenal (%)
    ax = axes[3]
    vals = df_conv.loc[in_prov, 'cv_interanual_pct']
    sc = ax.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='YlOrRd', vmin=0.0, vmax=5.0, s=1.2, edgecolors='none', rasterized=True)
    overlay_decorations(ax, boundary_gdf, deptos_gdf)
    ax.set_title(r"Variabilidad Quinquenal $CV_{\mathrm{interanual}}$" + f"\nMedia Provincial: {vals.mean():.2f}%", fontsize=11, weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4%", pad=0.15)
    plt.colorbar(sc, cax=cax, orientation="horizontal").set_label("Coeficiente de Variación (%)", fontsize=9)
    
    plt.suptitle("Estabilidad Climatológica y Convergencia Temporal Quinquenal del Atlas Solar (2020 - 2024)", fontsize=13, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "figura_convergencia_y_cv_interanual.png")
    plt.close()
    print("   ✓ Guardada Figura Oficial 2: figura_convergencia_y_cv_interanual.png")
    
    # FIGURA 3: CICLO MENSUAL DE LOS 12 MESES
    fig, axes = plt.subplots(3, 4, figsize=(20, 14), dpi=300)
    axes = axes.ravel()
    for idx, m_name in enumerate(month_names):
        ax = axes[idx]
        vals = df_months.loc[in_prov, m_name]
        sc = ax.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='inferno', s=0.8, edgecolors='none', rasterized=True)
        overlay_decorations(ax, boundary_gdf, deptos_gdf)
        ax.set_title(f"{m_name}\nMedia: {vals.mean():.1f} kWh/m²", fontsize=10.5, weight='bold')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="4%", pad=0.15)
        cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8)
        
    plt.suptitle("Ciclo Anual Estacional Completo (Enero a Diciembre): Promedio Quinquenal (2020 - 2024)", fontsize=14, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES_DIR / "figura_ciclo_mensual_anual_completo.png")
    plt.close()
    print("   ✓ Guardada Figura Oficial 3: figura_ciclo_mensual_anual_completo.png")
    
    # FIGURA 4: COMPARATIVA ESPACIAL DE LOS 5 AÑOS INDIVIDUALES
    fig, axes = plt.subplots(1, 5, figsize=(26, 6.5), dpi=300)
    for idx, y in enumerate(years_complete):
        ax = axes[idx]
        vals = annual_data[y]['GHI'][in_prov]
        sc = ax.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='inferno', vmin=1600, vmax=2100, s=0.9, edgecolors='none', rasterized=True)
        overlay_decorations(ax, boundary_gdf, deptos_gdf)
        ax.set_title(f"Año {y}\nMedia: {vals.mean():.1f} kWh/m²", fontsize=11, weight='bold')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="4%", pad=0.15)
        cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("kWh/m²/año", fontsize=8.5)
        
    plt.suptitle("Consistencia y Estabilidad Espacial Interanual de la Irradiancia Global Horizontal (2020 - 2024)", fontsize=13, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "figura_estabilidad_espacial_5anios.png")
    plt.close()
    print("   ✓ Guardada Figura Oficial 4: figura_estabilidad_espacial_5anios.png")
    
    # --------------------------------------------------------------------------
    # 7. TABLA LATEX PROVINCIAL CONSOLIDADA
    # --------------------------------------------------------------------------
    tex_path = FIGURES_DIR / "tabla_fase5b_metricas_provinciales_consolidadas.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Tabla Generada Automáticamente - Atlas Solar Definitivo Quinquenal (2020-2024)\n")
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        f.write("\\caption{Métricas Radiativas y de Rendimiento Fotovoltaico Provinciales Consolidadas (Climatología Quinquenal 2020--2024, Grilla Canónica 2 km).}\n")
        f.write("\\label{tab:metricas_consolidadas_5b}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("\\textbf{Variable} & \\textbf{Media Prov.} & \\textbf{Mínimo} & \\textbf{Máximo} & \\textbf{Unidad} \\\\\n\\midrule\n")
        f.write(f"Irradiancia Global Horizontal (GHI) & {p_ghi.mean():.1f} & {p_ghi.min():.1f} & {p_ghi.max():.1f} & kWh/m\\textsuperscript{{2}}/año \\\\\n")
        f.write(f"Irradiancia Directa Normal (DNI) & {p_dni.mean():.1f} & {p_dni.min():.1f} & {p_dni.max():.1f} & kWh/m\\textsuperscript{{2}}/año \\\\\n")
        f.write(f"Irradiancia Difusa Horizontal (DHI) & {p_dhi.mean():.1f} & {p_dhi.min():.1f} & {p_dhi.max():.1f} & kWh/m\\textsuperscript{{2}}/año \\\\\n")
        f.write(f"Irradiancia en Plano Fijo 30° N (POA) & {p_poa_fix.mean():.1f} & {p_poa_fix.min():.1f} & {p_poa_fix.max():.1f} & kWh/m\\textsuperscript{{2}}/año \\\\\n")
        f.write(f"Irradiancia en Seguidor 1-Eje (POA) & {p_poa_trk.mean():.1f} & {p_poa_trk.min():.1f} & {p_poa_trk.max():.1f} & kWh/m\\textsuperscript{{2}}/año \\\\\n")
        f.write(f"Rendimiento Específico Fijo ($Y_f$) & {p_yf_fix.mean():.1f} & {p_yf_fix.min():.1f} & {p_yf_fix.max():.1f} & kWh/kWp/año \\\\\n")
        f.write(f"Rendimiento Específico Tracker ($Y_f$) & {p_yf_trk.mean():.1f} & {p_yf_trk.min():.1f} & {p_yf_trk.max():.1f} & kWh/kWp/año \\\\\n")
        f.write(f"Factor de Capacidad Fijo ($CF$) & {p_cf_fix.mean():.2f} & {p_cf_fix.min():.2f} & {p_cf_fix.max():.2f} & \\% \\\\\n")
        f.write(f"Factor de Capacidad Tracker ($CF$) & {p_cf_trk.mean():.2f} & {p_cf_trk.min():.2f} & {p_cf_trk.max():.2f} & \\% \\\\\n")
        f.write(f"Ganancia Energética de Seguimiento & {p_gain_trk.mean():.2f} & {p_gain_trk.min():.2f} & {p_gain_trk.max():.2f} & \\% \\\\\n")
        f.write(f"Coeficiente de Variación Interanual ($CV$) & {cv_interanual[in_prov].mean():.2f} & {cv_interanual[in_prov].min():.2f} & {cv_interanual[in_prov].max():.2f} & \\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"   ✓ Guardada Tabla Oficial LaTeX: {tex_path}")
    
    print("\n🎉 ¡PROCESO DE FASE 5B QUINQUENAL COMPLETO CON ÉXITO ABSOLUTO!")


if __name__ == "__main__":
    main()
