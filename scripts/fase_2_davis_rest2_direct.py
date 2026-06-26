#!/usr/import/env python3
# -*- coding: utf-8 -*-

"""
Fase 2.4: Cálculo Directo de REST2 sobre las coordenadas y tiempos de la Red Davis (2022-2025)
Utiliza la base de datos completa de ERA5 y CAMS descargada para simular cielo claro.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import xarray as xr
import pvlib
from scipy.interpolate import RegularGridInterpolator

# Config
# ------------------------------------------------------------------------------
STATION_QC_FILE = Path("data/processed/stations/davis_qc_hourly.parquet")
ATM_DIR = Path("data/raw/atm")
OUT_FILE = Path("data/processed/stations/davis_qc_with_rest2.parquet")

DEFAULT_AOD_550 = 0.1
ALPHA = 1.3 

def load_era5_interpolator(nc_path):
    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    
    if 'valid_time' in ds.coords:
        times = ds.valid_time.values.astype("float64")
    else:
        times = ds.time.values.astype("float64")
        
    lats = ds.latitude.values
    lons = ds.longitude.values
    
    sp = ds['sp'].values
    tcwv = ds['tcwv'].values
    
    interp_sp = RegularGridInterpolator((times, lats, lons), sp, bounds_error=False, fill_value=None)
    interp_tcwv = RegularGridInterpolator((times, lats, lons), tcwv, bounds_error=False, fill_value=None)
    
    ds.close()
    return interp_sp, interp_tcwv

def load_cams_interpolator(nc_path):
    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    
    if 'valid_time' in ds.coords:
        times = ds.valid_time.values.astype("float64")
    else:
        times = ds.time.values.astype("float64")
        
    lats = ds.latitude.values
    lons = ds.longitude.values
    
    var_name = None
    for v in ['total_aerosol_optical_depth_550nm', 'aod550', 'omaod550']: 
        if v in ds.variables:
            var_name = v
            break
            
    if var_name is None:
        return None

    val = ds[var_name].values
    interp_aod = RegularGridInterpolator((times, lats, lons), val, bounds_error=False, fill_value=None)
    
    ds.close()
    return interp_aod

def main():
    print("🚀 Iniciando Simulación Directa de REST2 para Estaciones Davis...")
    
    if not STATION_QC_FILE.exists():
        print(f"❌ Error: No se encuentra {STATION_QC_FILE}")
        return
        
    df_st = pd.read_parquet(STATION_QC_FILE)
    print(f"📂 Dataset de estaciones cargado: {len(df_st)} registros.")
    
    # Asegurar orden temporal
    df_st = df_st.sort_index()
    
    # Obtener lista de meses presentes en los datos de las estaciones
    # df_st.index es DatetimeIndex en UTC
    df_st['year_month'] = df_st.index.strftime("%Y%m")
    unique_months = sorted(df_st['year_month'].unique())
    print(f"📅 Meses a procesar: {unique_months}")
    
    processed_dfs = []
    
    for ym in unique_months:
        print(f"⚡ Procesando mes: {ym}...")
        
        atm_file = ATM_DIR / f"era5_atm_{ym}.nc"
        cams_file = ATM_DIR / f"cams_aod_{ym}.nc"
        
        if not atm_file.exists():
            print(f"   ⚠️  Falta archivo ERA5 para el mes {ym}. Saltando.")
            continue
            
        use_cams = cams_file.exists()
        
        try:
            interp_sp, interp_tcwv = load_era5_interpolator(atm_file)
            interp_aod = load_cams_interpolator(cams_file) if use_cams else None
            
            # Filtrar registros de estaciones para este mes
            df_month = df_st[df_st['year_month'] == ym].copy()
            if df_month.empty:
                continue
                
            # Preparar puntos de interpolación: (tiempo_float, lat, lon)
            t_vals = df_month.index.values.astype("float64")
            pts = np.column_stack((t_vals, df_month['latitude'].values, df_month['longitude'].values))
            
            # Interpolación
            pressure_pa = interp_sp(pts)
            pwv_mm = interp_tcwv(pts)
            
            if interp_aod:
                aod550_val = interp_aod(pts)
                aod550_val = np.nan_to_num(aod550_val, nan=DEFAULT_AOD_550)
            else:
                aod550_val = np.full(len(df_month), DEFAULT_AOD_550)
                
            # Conversión de unidades para Solis/REST2
            pwv_cm = pwv_mm / 10.0
            aod700 = aod550_val * (550 / 700) ** ALPHA
            
            # Calcular elevación solar para cada punto
            # Para mayor precisión y velocidad, usamos pvlib por estación dentro del mes
            df_month['ghi_cs'] = np.nan
            df_month['dni_cs'] = np.nan
            df_month['dhi_cs'] = np.nan
            df_month['cosZ'] = np.nan
            df_month['zenith'] = np.nan
            
            for est, df_est in df_month.groupby('estacion'):
                idx = df_est.index
                lat = df_est['latitude'].iloc[0]
                lon = df_est['longitude'].iloc[0]
                
                solpos = pvlib.solarposition.get_solarposition(
                    time=idx,
                    latitude=lat,
                    longitude=lon
                )
                
                elev = 90.0 - solpos['zenith']
                cos_z = np.cos(np.radians(solpos['zenith'].values))
                
                # Obtener los índices de máscara booleanos como array de numpy
                loc_mask = (df_month['estacion'] == est).values
                
                # Alinear las series de pandas usando el mismo índice temporal
                pwv_series = pd.Series(pwv_cm[loc_mask], index=idx)
                press_series = pd.Series(pressure_pa[loc_mask], index=idx)
                aod_series = pd.Series(aod700[loc_mask], index=idx)
                
                # Ejecutar Solis
                cs = pvlib.clearsky.simplified_solis(
                    apparent_elevation=elev,
                    precipitable_water=pwv_series,
                    pressure=press_series,
                    aod700=aod_series
                )
                
                # Guardar en df_month
                df_month.loc[df_month['estacion'] == est, 'ghi_cs'] = cs['ghi'].values
                df_month.loc[df_month['estacion'] == est, 'dni_cs'] = cs['dni'].values
                df_month.loc[df_month['estacion'] == est, 'dhi_cs'] = cs['dhi'].values
                df_month.loc[df_month['estacion'] == est, 'cosZ'] = cos_z
                df_month.loc[df_month['estacion'] == est, 'zenith'] = solpos['zenith'].values
                
            processed_dfs.append(df_month)
            print(f"   ✅ Mes {ym} procesado correctamente. Registros: {len(df_month)}")
            
        except Exception as e:
            print(f"   ❌ Error al procesar el mes {ym}: {e}")
            
    if processed_dfs:
        print("\n💾 Consolidando todos los meses...")
        final_df = pd.concat(processed_dfs)
        # Eliminar columna temporal
        final_df.drop(columns=['year_month'], inplace=True)
        
        # Guardar
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(OUT_FILE, index=True)
        print(f"✅ ¡Simulación finalizada! Guardado en {OUT_FILE}")
    else:
        print("⚠️ No se procesó ningún dato.")

if __name__ == "__main__":
    main()
