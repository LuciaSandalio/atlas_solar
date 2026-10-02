#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 5: Atlas Solar Preliminar de Córdoba (Años Completos 2020, 2021, 2023 + Diagnóstico 2024)
=============================================================================================
Convenio ATLAS II — Provincia de Córdoba (Componente Solar)

1. Regla estricta de agregación anual:
   Utiliza exclusivamente los años completos: 2020 (366 d), 2021 (365 d) y 2023 (365 d).
   H_y(x, y) = sum_t GHI_y(x, y, t) * Delta t
   H_prelim(x, y) = (H_2020 + H_2021 + H_2023) / 3

2. Cadena física completa por celda (59,837 celdas a 2 km):
   - Ineichen Clearsky con turbidez Linke y altimetría topográfica.
   - Brillo de cielo claro Tarpley B0 y contraste nuboso Fr.
   - Inferencia de Kt mediante HybridBDJPT_HGB (Opción C: mean(t, t+1)).
   - Descomposición radiativa Erbs et al. (1982) -> DNI, DHI.
   - Transposición Perez et al. (1990):
     * Configuración A: Plano Fijo 30° Norte (azimut 0°).
     * Configuración B: Seguidor 1-eje horizontal N-S con backtracking (GCR=0.35, max 60°).
   - Modelo térmico NOCT (45°C) y modelo paramétrico PVWatts (DC + Inversor AC + 10% BOS).

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

4. Control de Estabilidad Espacial (1 enero -- 20 agosto 2020, 2021, 2023, 2024):
   - Cuatro mapas comparables del período común y matriz de correlación espacial.
   - Promedios mensuales para enero–julio (4 realizaciones).

5. Prueba de Convergencia Temporal y Estabilidad Interanual:
   - Acumulados sucesivos: M_2020, M_20-21, M_20-21-23.
   - Incrementos relativos: Delta_2 (1 -> 2 años) y Delta_3 (2 -> 3 años).
   - Coeficiente de variación interanual: CV_interanual (%).
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
PRELIM_DIR = READY_DIR / "atlas_preliminar_2km"
PRELIM_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = ROOT_DIR / "reportes" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("reportes/figures")

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
    g_alt = np.load(READY_DIR / 'altitude_grid_2km.npy')
    g_pres = np.load(READY_DIR / 'pressure_grid_2km.npy')
    g_linke = np.load(READY_DIR / 'linke_turbidity_grid_2km.npy')
    g_mod_C = joblib.load(READY_DIR / 'b0_model' / 'model_hybrid_bdjpt_hgb_optionC.pkl')
    g_n_cells = len(g_alt)


def process_single_day(fpath_str):
    fpath = Path(fpath_str)
    day_str = fpath.stem.replace('reflectancia_', '')
    dt = pd.to_datetime(day_str)
    doy = int(dt.dayofyear)
    month_idx = int(dt.month) - 1
    
    try:
        df_day = pd.read_parquet(fpath)
    except Exception as e:
        print(f"⚠️ Error leyendo {fpath}: {e}. Saltando.")
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
        # Opción C: promedio temporal continuo con la hora siguiente si es consecutiva
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
        
        # Ineichen Clearsky
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
        
        # Inferencia de Kt con HybridBDJPT_HGB
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
        
        # Transposición Tracker 1-eje N-S con Backtracking (GCR 0.35)
        trk = pvlib.tracking.singleaxis(apparent_zenith=app_zen, solar_azimuth=az, axis_tilt=0.0, axis_azimuth=0.0, max_angle=60.0, backtrack=True, gcr=0.35)
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


def overlay_decorations(ax, boundary_gdf, deptos_gdf=None):
    if deptos_gdf is not None:
        deptos_gdf.boundary.plot(ax=ax, color="#AAAAAA", linewidth=0.5, zorder=2)
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color="#222222", linewidth=1.1, zorder=3)
    for name, (clon, clat) in CIUDADES_CORDOBA.items():
        ax.plot(clon, clat, 'ko', markersize=3.0, zorder=5)
        ax.text(clon + 0.06, clat - 0.04, name, fontsize=7.5, weight='bold',
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8), zorder=6)
    ax.set_xlim(-65.8, -61.6)
    ax.set_ylim(-35.2, -29.4)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    print("=" * 80)
    print("🚀 FASE 5: ATLAS SOLAR PRELIMINAR DE CÓRDOBA (2020, 2021, 2023 + DIAGNÓSTICO 2024)")
    print("=" * 80)
    
    # 1. Cargar Grilla Provincial
    print("\n📂 1. Cargando grilla provincial canónica de 2 km...")
    grid = pd.read_parquet(READY_DIR / "cordoba_grid_2km_regions.parquet")
    n_cells = len(grid)
    lats = grid['lat'].values
    lons = grid['lon'].values
    in_prov = grid['in_province'].values.astype(bool)
    print(f"   • Total celdas: {n_cells:,} (En territorio provincial: {in_prov.sum():,})")
    
    # Cargar límites territoriales
    boundary_gdf = gpd.read_file(DATA_DIR / "cordoba_boundary.geojson") if (DATA_DIR / "cordoba_boundary.geojson").exists() else None
    deptos_gdf = gpd.read_file(ROOT_DIR / "shapes_cba" / "cba_deptos.shp") if (ROOT_DIR / "shapes_cba" / "cba_deptos.shp").exists() else None
    
    # 2. Filtrar archivos por año
    years_to_process = ['2020', '2021', '2023', '2024']
    all_files = sorted(glob.glob(str(PROCESSED_2KM_DIR / "reflectancia_*.parquet")))
    
    year_files = {}
    for y in years_to_process:
        if y == '2024':
            # Solo hasta el 20 de agosto inclusive
            flist = [f for f in all_files if Path(f).stem.replace('reflectancia_', '') >= '20240101' and Path(f).stem.replace('reflectancia_', '') <= '20240820']
        else:
            flist = [f for f in all_files if Path(f).stem.replace('reflectancia_', '').startswith(y)]
        year_files[y] = flist
        print(f"   • Año {y}: {len(flist)} archivos encontrados.")
        
    # Inicializar estructuras por año
    annual_data = {}
    monthly_data = {y: np.zeros((n_cells, 12), dtype=np.float32) for y in years_to_process}
    partial_ghi_jan_aug20 = {}
    
    # Usar 6 workers para paralelismo balanceado
    num_workers = min(6, cpu_count())
    print(f"\n⚡ 2. Ejecutando inferencia masiva y transposición con {num_workers} procesos paralelos...")
    
    t_start = time.time()
    
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        for y in years_to_process:
            flist = year_files[y]
            print(f"\n--- Procesando Año {y} ({len(flist)} días) ---")
            
            ann_file = PRELIM_DIR / f"atlas_anual_{y}.parquet"
            monthly_file = PRELIM_DIR / f"atlas_monthly_{y}.npy"
            part_file = PRELIM_DIR / f"atlas_partial_{y}.npy"
            if (ann_file.exists() or y == '2024') and monthly_file.exists() and part_file.exists():
                print(f"🔄 Año {y} ya procesado previamente. Cargando checkpoint...")
                if y in ['2020', '2021', '2023']:
                    df_y = pd.read_parquet(ann_file)
                    annual_data[y] = {
                        'GHI': df_y['ghi_kwh_m2'].values,
                        'DNI': df_y['dni_kwh_m2'].values,
                        'DHI': df_y['dhi_kwh_m2'].values,
                        'POA_fixed': df_y['poa_fixed_kwh_m2'].values,
                        'POA_tracker': df_y['poa_tracker_kwh_m2'].values,
                        'Yf_fixed': df_y['yf_fixed_kwh_kwp'].values,
                        'Yf_tracker': df_y['yf_tracker_kwh_kwp'].values
                    }
                monthly_data[y] = np.load(monthly_file)
                partial_ghi_jan_aug20[y] = np.load(part_file)
                continue
            
            y_ghi = np.zeros(n_cells, dtype=np.float32)
            y_dni = np.zeros(n_cells, dtype=np.float32)
            y_dhi = np.zeros(n_cells, dtype=np.float32)
            y_poa_fix = np.zeros(n_cells, dtype=np.float32)
            y_poa_trk = np.zeros(n_cells, dtype=np.float32)
            y_yf_fix = np.zeros(n_cells, dtype=np.float32)
            y_yf_trk = np.zeros(n_cells, dtype=np.float32)
            y_part_ghi = np.zeros(n_cells, dtype=np.float32)
            
            t_y0 = time.time()
            res_iter = pool.imap_unordered(process_single_day, flist, chunksize=4)
            
            for idx, r in enumerate(res_iter):
                if r is None:
                    continue
                day_str, m_idx, d_ghi, d_dni, d_dhi, d_p_fix, d_p_trk, d_y_fix, d_y_trk = r
                
                # Conversión de Wh a kWh para acumuladores
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
                
                # Control Jan 1 - Aug 20
                if day_str[4:] <= '0820':
                    y_part_ghi += kwh_ghi
                    
                if (idx + 1) % 50 == 0 or (idx + 1) == len(flist):
                    print(f"   [{idx + 1:3d}/{len(flist):3d}] procesados | Tiempo: {time.time()-t_y0:.1f} s | Media GHI diaria: {kwh_ghi.mean():.2f} kWh/m²")
                    
            partial_ghi_jan_aug20[y] = y_part_ghi
            
            # Guardar matrices mensuales y parciales para reanudación instantánea
            np.save(PRELIM_DIR / f"atlas_monthly_{y}.npy", monthly_data[y])
            np.save(PRELIM_DIR / f"atlas_partial_{y}.npy", y_part_ghi)
            
            if y in ['2020', '2021', '2023']:
                annual_data[y] = {
                    'GHI': y_ghi,
                    'DNI': y_dni,
                    'DHI': y_dhi,
                    'POA_fixed': y_poa_fix,
                    'POA_tracker': y_poa_trk,
                    'Yf_fixed': y_yf_fix,
                    'Yf_tracker': y_yf_trk
                }
                # Guardar Parquet del año
                df_y = pd.DataFrame({
                    'cell_id': grid['cell_id'],
                    'lat': lats,
                    'lon': lons,
                    'in_province': in_prov,
                    'ghi_kwh_m2': y_ghi,
                    'dni_kwh_m2': y_dni,
                    'dhi_kwh_m2': y_dhi,
                    'poa_fixed_kwh_m2': y_poa_fix,
                    'poa_tracker_kwh_m2': y_poa_trk,
                    'yf_fixed_kwh_kwp': y_yf_fix,
                    'yf_tracker_kwh_kwp': y_yf_trk
                })
                df_y.to_parquet(PRELIM_DIR / f"atlas_anual_{y}.parquet", index=False)
                print(f"✓ Año {y} completado y guardado en {PRELIM_DIR / f'atlas_anual_{y}.parquet'}")
                print(f"   • GHI medio provincial: {y_ghi[in_prov].mean():.1f} kWh/m²/año | Yf Fijo: {y_yf_fix[in_prov].mean():.1f} | Yf Tracker: {y_yf_trk[in_prov].mean():.1f}")
                
    print(f"\n✅ Inferencia completada en {(time.time()-t_start)/60:.2f} minutos.")
    
    # --------------------------------------------------------------------------
    # 3. CONSOLIDACIÓN DE LOS 10 MAPAS OFICIALES (AÑOS COMPLETOS 2020, 2021, 2023)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 3. CALCULANDO ATLAS ANUAL MEDIO PRELIMINAR (2020, 2021, 2023)")
    print("=" * 80)
    
    ghi_mean = (annual_data['2020']['GHI'] + annual_data['2021']['GHI'] + annual_data['2023']['GHI']) / 3.0
    dni_mean = (annual_data['2020']['DNI'] + annual_data['2021']['DNI'] + annual_data['2023']['DNI']) / 3.0
    dhi_mean = (annual_data['2020']['DHI'] + annual_data['2021']['DHI'] + annual_data['2023']['DHI']) / 3.0
    poa_fix_mean = (annual_data['2020']['POA_fixed'] + annual_data['2021']['POA_fixed'] + annual_data['2023']['POA_fixed']) / 3.0
    poa_trk_mean = (annual_data['2020']['POA_tracker'] + annual_data['2021']['POA_tracker'] + annual_data['2023']['POA_tracker']) / 3.0
    yf_fix_mean = (annual_data['2020']['Yf_fixed'] + annual_data['2021']['Yf_fixed'] + annual_data['2023']['Yf_fixed']) / 3.0
    yf_trk_mean = (annual_data['2020']['Yf_tracker'] + annual_data['2021']['Yf_tracker'] + annual_data['2023']['Yf_tracker']) / 3.0
    
    cf_fix_mean = (yf_fix_mean / 8760.0) * 100.0
    cf_trk_mean = (yf_trk_mean / 8760.0) * 100.0
    gain_trk_mean = ((yf_trk_mean - yf_fix_mean) / np.maximum(yf_fix_mean, 1e-3)) * 100.0
    
    # Crear DataFrame oficial de los 10 mapas
    df_atlas_10 = pd.DataFrame({
        'cell_id': grid['cell_id'],
        'lat': lats,
        'lon': lons,
        'in_province': in_prov,
        'region': grid['region'],
        'ghi_anual_kwh_m2': np.round(ghi_mean, 1),
        'dni_anual_kwh_m2': np.round(dni_mean, 1),
        'dhi_anual_kwh_m2': np.round(dhi_mean, 1),
        'poa_fixed_kwh_m2': np.round(poa_fix_mean, 1),
        'poa_tracker_kwh_m2': np.round(poa_trk_mean, 1),
        'yf_fixed_kwh_kwp': np.round(yf_fix_mean, 1),
        'yf_tracker_kwh_kwp': np.round(yf_trk_mean, 1),
        'cf_fixed_pct': np.round(cf_fix_mean, 2),
        'cf_tracker_pct': np.round(cf_trk_mean, 2),
        'ganancia_tracker_pct': np.round(gain_trk_mean, 2)
    })
    
    df_atlas_10.to_parquet(PRELIM_DIR / "atlas_anual_2020_2021_2023_medio.parquet", index=False)
    print(f"✓ Base de datos oficial guardada en {PRELIM_DIR / 'atlas_anual_2020_2021_2023_medio.parquet'}")
    
    # Resumen provincial por terminal
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
    
    print("\nRESUMEN TERRITORIAL PROVINCIAL (Promedio 2020, 2021, 2023):")
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
    # 4. CONTROL DE ESTABILIDAD ESPACIAL (1 ENERO -- 20 AGOSTO 2020-2024)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔬 4. CONTROL DE ESTABILIDAD ESPACIAL (1 ENE -- 20 AGOSTO: 2020, 2021, 2023, 2024)")
    print("=" * 80)
    
    df_part = pd.DataFrame({
        'cell_id': grid['cell_id'],
        'lat': lats,
        'lon': lons,
        'in_province': in_prov,
        'GHI_JanAug20_2020': partial_ghi_jan_aug20['2020'],
        'GHI_JanAug20_2021': partial_ghi_jan_aug20['2021'],
        'GHI_JanAug20_2023': partial_ghi_jan_aug20['2023'],
        'GHI_JanAug20_2024': partial_ghi_jan_aug20['2024']
    })
    df_part.to_parquet(PRELIM_DIR / "atlas_estabilidad_parcial_jan_aug20.parquet", index=False)
    
    # Matriz de correlación espacial provincial
    corr_matrix = df_part.loc[in_prov, ['GHI_JanAug20_2020', 'GHI_JanAug20_2021', 'GHI_JanAug20_2023', 'GHI_JanAug20_2024']].corr()
    print("Matriz de Correlación Espacial (Período 1 Ene - 20 Ago):")
    print(corr_matrix.to_string())
    corr_matrix.to_csv(RESULTS_DIR / "correlacion_espacial_1ene_20ago.csv")
    
    # --------------------------------------------------------------------------
    # 5. CONVERGENCIA TEMPORAL Y ESTABILIDAD INTERANUAL
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📈 5. ANÁLISIS DE CONVERGENCIA TEMPORAL Y COEFICIENTE DE VARIACIÓN INTERANUAL")
    print("=" * 80)
    
    M_20 = annual_data['2020']['GHI']
    M_20_21 = (annual_data['2020']['GHI'] + annual_data['2021']['GHI']) / 2.0
    M_20_21_23 = (annual_data['2020']['GHI'] + annual_data['2021']['GHI'] + annual_data['2023']['GHI']) / 3.0
    
    delta_2 = 100.0 * (M_20_21 - M_20) / np.maximum(M_20, 1.0)
    delta_3 = 100.0 * (M_20_21_23 - M_20_21) / np.maximum(M_20_21, 1.0)
    
    # Coeficiente de variación interanual entre los 3 años completos
    arr_3y = np.column_stack([annual_data['2020']['GHI'], annual_data['2021']['GHI'], annual_data['2023']['GHI']])
    sigma_3y = np.std(arr_3y, axis=1)
    mu_3y = np.mean(arr_3y, axis=1)
    cv_interanual = 100.0 * (sigma_3y / np.maximum(mu_3y, 1.0))
    
    df_conv = pd.DataFrame({
        'cell_id': grid['cell_id'],
        'lat': lats,
        'lon': lons,
        'in_province': in_prov,
        'M_2020': M_20,
        'M_2020_2021': M_20_21,
        'M_2020_2021_2023': M_20_21_23,
        'delta_2_pct': delta_2,
        'delta_3_pct': delta_3,
        'cv_interanual_pct': cv_interanual
    })
    df_conv.to_parquet(PRELIM_DIR / "atlas_convergencia_temporal.parquet", index=False)
    
    print(f"Convergencia del Atlas Provincial (Celdas Provinciales):")
    print(f"   • Delta 2 (1 -> 2 años): Media absoluta = {np.abs(delta_2[in_prov]).mean():.2f}% | Rango: [{delta_2[in_prov].min():.2f}%, {delta_2[in_prov].max():.2f}%]")
    print(f"   • Delta 3 (2 -> 3 años): Media absoluta = {np.abs(delta_3[in_prov]).mean():.2f}% | Rango: [{delta_3[in_prov].min():.2f}%, {delta_3[in_prov].max():.2f}%]")
    print(f"   • CV Interanual Medio : {cv_interanual[in_prov].mean():.2f}% | Rango: [{cv_interanual[in_prov].min():.2f}%, {cv_interanual[in_prov].max():.2f}%]")
    
    # --------------------------------------------------------------------------
    # 6. PROMEDIOS MENSUALES (ENERO -- JULIO, 4 REALIZACIONES)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📅 6. PROMEDIOS MENSUALES (ENERO -- JULIO: 4 AÑOS)")
    print("=" * 80)
    
    month_names = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio']
    monthly_means = {}
    for m in range(7):
        m_mean = (monthly_data['2020'][:, m] + monthly_data['2021'][:, m] + monthly_data['2023'][:, m] + monthly_data['2024'][:, m]) / 4.0
        monthly_means[month_names[m]] = m_mean
        print(f"   • {month_names[m]:<10}: {m_mean[in_prov].mean():.1f} kWh/m²/mes")
        
    df_months = pd.DataFrame({'cell_id': grid['cell_id'], 'lat': lats, 'lon': lons, 'in_province': in_prov, **monthly_means})
    df_months.to_parquet(PRELIM_DIR / "atlas_promedios_mensuales_ene_jul.parquet", index=False)
    
    # --------------------------------------------------------------------------
    # 7. GENERACIÓN DE FIGURAS CARTOGRÁFICAS DE ALTA RESOLUCIÓN
    # --------------------------------------------------------------------------
    print("\n🎨 7. Generando cartografía temática oficial en alta resolución (300 DPI)...")
    
    # FIGURA 1: Los 10 Mapas Oficiales Anuales Medios
    fig, axes = plt.subplots(2, 5, figsize=(24, 11), dpi=300)
    axes = axes.ravel()
    
    map_configs = [
        ('ghi_anual_kwh_m2', '1. GHI Anual Medio\n(kWh/m²/año)', 'inferno', p_ghi.min(), p_ghi.max()),
        ('dni_anual_kwh_m2', '2. DNI Anual Medio\n(kWh/m²/año)', 'magma', p_dni.min(), p_dni.max()),
        ('dhi_anual_kwh_m2', '3. DHI Anual Medio\n(kWh/m²/año)', 'cividis', p_dhi.min(), p_dhi.max()),
        ('poa_fixed_kwh_m2', '4. POA Fijo 30° Norte\n(kWh/m²/año)', 'plasma', p_poa_fix.min(), p_poa_fix.max()),
        ('poa_tracker_kwh_m2', '5. POA Tracker 1-Eje\n(kWh/m²/año)', 'plasma', p_poa_trk.min(), p_poa_trk.max()),
        ('yf_fixed_kwh_kwp', '6. Rendimiento Yf Fijo\n(kWh/kWp/año)', 'viridis', p_yf_fix.min(), p_yf_fix.max()),
        ('yf_tracker_kwh_kwp', '7. Rendimiento Yf Tracker\n(kWh/kWp/año)', 'viridis', p_yf_trk.min(), p_yf_trk.max()),
        ('cf_fixed_pct', '8. Factor Capacidad Fijo\n(%)', 'turbo', p_cf_fix.min(), p_cf_fix.max()),
        ('cf_tracker_pct', '9. Factor Capacidad Tracker\n(%)', 'turbo', p_cf_trk.min(), p_cf_trk.max()),
        ('ganancia_tracker_pct', '10. Ganancia Tracker s/ Fijo\n(%)', 'YlGnBu', p_gain_trk.min(), p_gain_trk.max())
    ]
    
    for idx, (col, title, cmap, vmin, vmax) in enumerate(map_configs):
        ax = axes[idx]
        vals = df_atlas_10.loc[in_prov, col]
        sc = ax.scatter(lons[in_prov], lats[in_prov], c=vals, cmap=cmap, vmin=vmin, vmax=vmax, s=0.8, edgecolors='none', rasterized=True)
        overlay_decorations(ax, boundary_gdf, deptos_gdf)
        ax.set_title(title, fontsize=10.5, weight='bold', pad=8)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="4%", pad=0.15)
        cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8)
        
    plt.suptitle("Atlas Solar Preliminar de la Provincia de Córdoba\nPromedio Territorial Multianual Basado en Años Completos (2020, 2021 y 2023) — Grilla Canónica 2 km", fontsize=15, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES_DIR / "figura_atlas_preliminar_10_mapas.png")
    fig.savefig(ARTIFACT_DIR / "figura_atlas_preliminar_10_mapas.png")
    plt.close()
    print("   ✓ Guardada Figura 1: figura_atlas_preliminar_10_mapas.png")
    
    # FIGURA 2: Control de Estabilidad Espacial (1 de enero -- 20 de agosto: 2020, 2021, 2023, 2024)
    fig, axes = plt.subplots(1, 4, figsize=(20, 6.5), dpi=300)
    p_part_all = np.concatenate([df_part.loc[in_prov, f'GHI_JanAug20_{y}'] for y in years_to_process])
    vmin_part, vmax_part = np.percentile(p_part_all, 1), np.percentile(p_part_all, 99)
    
    for idx, y in enumerate(years_to_process):
        ax = axes[idx]
        vals = df_part.loc[in_prov, f'GHI_JanAug20_{y}']
        sc = ax.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='inferno', vmin=vmin_part, vmax=vmax_part, s=1.2, edgecolors='none', rasterized=True)
        overlay_decorations(ax, boundary_gdf, deptos_gdf)
        ax.set_title(f"GHI (1 Ene - 20 Ago) {y}\nMedia: {vals.mean():.1f} kWh/m²", fontsize=11, weight='bold')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="4%", pad=0.15)
        cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8)
        
    plt.suptitle("Control de Estabilidad Territorial del Recurso Solar (Período Idéntico 1 Ene -- 20 Ago: 2020, 2021, 2023, 2024)", fontsize=13, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "figura_estabilidad_espacial_jan_aug20.png")
    fig.savefig(ARTIFACT_DIR / "figura_estabilidad_espacial_jan_aug20.png")
    plt.close()
    print("   ✓ Guardada Figura 2: figura_estabilidad_espacial_jan_aug20.png")
    
    # FIGURA 3: Convergencia Temporal y Variabilidad Interanual
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), dpi=300)
    
    # Delta 2 (1 -> 2 años)
    ax = axes[0]
    vals = df_conv.loc[in_prov, 'delta_2_pct']
    sc = ax.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='RdBu_r', vmin=-6.0, vmax=6.0, s=1.2, edgecolors='none', rasterized=True)
    overlay_decorations(ax, boundary_gdf, deptos_gdf)
    ax.set_title(r"$\Delta_2$ (1 $\to$ 2 Años: $M_{20-21}$ vs $M_{20}$)" + f"\nMedia Absoluta: {np.abs(vals).mean():.2f}%", fontsize=11, weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4%", pad=0.15)
    cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label("% de variación", fontsize=9)
    
    # Delta 3 (2 -> 3 años)
    ax = axes[1]
    vals = df_conv.loc[in_prov, 'delta_3_pct']
    sc = ax.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='RdBu_r', vmin=-6.0, vmax=6.0, s=1.2, edgecolors='none', rasterized=True)
    overlay_decorations(ax, boundary_gdf, deptos_gdf)
    ax.set_title(r"$\Delta_3$ (2 $\to$ 3 Años: $M_{20-21-23}$ vs $M_{20-21}$)" + f"\nMedia Absoluta: {np.abs(vals).mean():.2f}%", fontsize=11, weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4%", pad=0.15)
    cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label("% de variación", fontsize=9)
    
    # CV Interanual (%)
    ax = axes[2]
    vals = df_conv.loc[in_prov, 'cv_interanual_pct']
    sc = ax.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='YlOrRd', vmin=0.0, vmax=6.0, s=1.2, edgecolors='none', rasterized=True)
    overlay_decorations(ax, boundary_gdf, deptos_gdf)
    ax.set_title(r"Variabilidad Interanual $CV_{\mathrm{interanual}}$" + f"\nMedia Provincial: {vals.mean():.2f}%", fontsize=11, weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4%", pad=0.15)
    cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label("Coeficiente de Variación (%)", fontsize=9)
    
    plt.suptitle("Evaluación de Convergencia Temporal y Estabilidad Climatológica del Atlas Solar", fontsize=13, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "figura_convergencia_y_cv_interanual.png")
    fig.savefig(ARTIFACT_DIR / "figura_convergencia_y_cv_interanual.png")
    plt.close()
    print("   ✓ Guardada Figura 3: figura_convergencia_y_cv_interanual.png")
    
    # FIGURA 4: Ciclo Mensual (Enero -- Julio, 4 realizaciones)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5), dpi=300)
    axes = axes.ravel()
    
    for idx, m_name in enumerate(month_names):
        ax = axes[idx]
        vals = df_months.loc[in_prov, m_name]
        sc = ax.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='inferno', s=0.9, edgecolors='none', rasterized=True)
        overlay_decorations(ax, boundary_gdf, deptos_gdf)
        ax.set_title(f"{m_name}\nMedia: {vals.mean():.1f} kWh/m²", fontsize=10.5, weight='bold')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="4%", pad=0.15)
        cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8)
        
    axes[7].axis('off') # Vacío
    plt.suptitle("Ciclo Estacional Medio Provincial (Enero -- Julio): Promedio 4 Años (2020, 2021, 2023, 2024)", fontsize=14, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES_DIR / "figura_ciclo_mensual_ene_jul.png")
    fig.savefig(ARTIFACT_DIR / "figura_ciclo_mensual_ene_jul.png")
    plt.close()
    print("   ✓ Guardada Figura 4: figura_ciclo_mensual_ene_jul.png")
    
    print("\n🎉 ¡PROCESO COMPLETO Y EXITOSO! TODOS LOS MAPAS Y ANÁLISIS GENERADOS.")


if __name__ == "__main__":
    main()
