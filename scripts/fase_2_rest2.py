#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fase 2.2: Modelo Clear-Sky (REST2 / Simplified Solis)

Input:
  - Solar Geometry: data/ready/features_cell_hour/features_YYYYMMDD.parquet
  - Atmosphere: data/raw/atm/era5_atm_YYYYMM.nc (ERA5: sp, tcwv, tco3)
  
Output:
  - Clear Sky GHI: data/ready/clearsky_rest2/cs_YYYYMMDD.parquet
"""

import pandas as pd
import numpy as np
import xarray as xr
import pvlib
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import glob
import os

# Config
# ------------------------------------------------------------------------------
INPUT_FEAT_DIR = Path("data/ready/features_cell_hour")
ATM_DIR = Path("data/raw/atm")
OUT_DIR = Path("data/ready/clearsky_rest2")

# Default AOD @ 550nm (if not available via CAMS)
DEFAULT_AOD_550 = 0.1
# Angstrom exponent for conversion to 700nm (Standard approximation)
ALPHA = 1.3 

def load_era5_interpolator(nc_path):
    """
    Loads ERA5 NetCDF and returns an interpolator function
    for (time, lat, lon) -> (sp, tcwv, tco3).
    """
    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    
    # Coordinates
    # ERA5 CDS often uses 'valid_time'
    if 'valid_time' in ds.coords:
        times = ds.valid_time.values.astype("float64")
    else:
        times = ds.time.values.astype("float64")
        
    lats = ds.latitude.values
    lons = ds.longitude.values
    
    # Variables
    # sp: Surface Pressure (Pa). Convert to Pascal if not already? It is Pa in ERA5.
    # tcwv: Total Column Water Vapour (kg/m^2 = mm). Solis needs cm.
    # tco3: Total Column Ozone (kg/m^2). Solis doesn't use Ozone directly??
    # Simplified Solis params: precipitable_water, pressure, aod700.
    
    sp = ds['sp'].values
    tcwv = ds['tcwv'].values
    
    # Create Interpolators
    # (time, lat, lon)
    # Bounds handling: fill_value=None (extrapolate) or bounds_error=False
    interp_sp = RegularGridInterpolator((times, lats, lons), sp, bounds_error=False, fill_value=None)
    interp_tcwv = RegularGridInterpolator((times, lats, lons), tcwv, bounds_error=False, fill_value=None)
    
    ds.close()
    return interp_sp, interp_tcwv

def load_cams_interpolator(nc_path):
    """
    Loads CAMS NetCDF and returns an interpolator for AOD.
    """
    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    
    # Coordinate check
    if 'valid_time' in ds.coords:
        times = ds.valid_time.values.astype("float64")
    else:
        times = ds.time.values.astype("float64")
        
    lats = ds.latitude.values
    lons = ds.longitude.values
    
    # Variable: total_aerosol_optical_depth_550nm
    # Name might vary ('aod550', 'omaod550', etc). Check typical CAMS vars.
    # Usually `total_aerosol_optical_depth_550nm` or similar. 
    # Let's try to detect or use the one present.
    var_name = None
    for v in ['total_aerosol_optical_depth_550nm', 'aod550', 'tco3', 'omaod550']: 
        if v in ds.variables:
            var_name = v
            break
            
    if var_name is None:
        # Fallback if we can't find it easily, print vars
        print(f"⚠️  CAMS Variable not found in {list(ds.variables)}")
        return None

    val = ds[var_name].values
    
    # Interpolator
    interp_aod = RegularGridInterpolator((times, lats, lons), val, bounds_error=False, fill_value=None)
    
    ds.close()
    return interp_aod

def process_day(feat_file):
    date_str = feat_file.stem.split("_")[1] # features_20200101
    year_month = date_str[:6] # 202001
    
    # Load Features
    df = pd.read_parquet(feat_file)
    if df.empty:
        return
        
    # Load ATM Data (ERA5)
    atm_file = ATM_DIR / f"era5_atm_{year_month}.nc"
    if not atm_file.exists():
        print(f"⚠️  Missing ATM file: {atm_file}")
        return

    # Load CAMS Data (AOD)
    cams_file = ATM_DIR / f"cams_aod_{year_month}.nc"
    use_cams = cams_file.exists()
    
    interp_sp, interp_tcwv = load_era5_interpolator(atm_file)
    
    interp_aod = None
    if use_cams:
        interp_aod = load_cams_interpolator(cams_file)

    
    # Prepare Interpolation Points
    # Format: (time_float, lat, lon)
    # df['time_round'] is datetime64[ns, UTC]. Convert to float/int
    t_vals = df['time_round'].values.astype("float64")
    pts = np.column_stack((t_vals, df['lat'].values, df['lon'].values))
    
    # Interpolate ERA5
    pressure_pa = interp_sp(pts)
    pwv_mm = interp_tcwv(pts)
    
    # Interpolate CAMS AOD
    if interp_aod:
        aod550_val = interp_aod(pts)
        # Handle missings/nans if any
        aod550_val = np.nan_to_num(aod550_val, nan=DEFAULT_AOD_550)
    else:
        aod550_val = DEFAULT_AOD_550
    
    # Convert Units for Solis
    # precipitable_water: cm (ERA5 is mm) => / 10
    pwv_cm = pwv_mm / 10.0
    
    # AOD 700nm conversion
    aod700 = aod550_val * (550 / 700) ** ALPHA
    
    # Run Model
    # simplified_solis(apparent_elevation, aod700, precipitable_water, pressure)
    # df has 'zenith_deg'. Elevation = 90 - Zenith.
    elevation = 90 - df['zenith_deg']
    
    cs = pvlib.clearsky.simplified_solis(
        apparent_elevation=elevation,
        precipitable_water=pwv_cm,
        pressure=pressure_pa,
        aod700=aod700
    )
    
    # Result is a DataFrame with 'ghi', 'dni', 'dhi'
    df_out = df[['cell_id', 'time_round']].copy()
    df_out['ghi_cs'] = cs['ghi']
    df_out['dni_cs'] = cs['dni']
    df_out['dhi_cs'] = cs['dhi']
    
    # Save
    out_file = OUT_DIR / f"cs_{date_str}.parquet"
    print(f"✅ Writing {out_file} (Rows: {len(df_out)}) [CAMS={use_cams}]")
    df_out.to_parquet(out_file, index=False)

def main():
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Run REST2 Clear-Sky Model")
    parser.add_argument("--start", help="YYYY-MM-DD")
    parser.add_argument("--end", help="YYYY-MM-DD")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    files = sorted(list(INPUT_FEAT_DIR.glob("features_*.parquet")))
    print(f"Found {len(files)} total feature files.")
    
    # Date Filtering
    start_dt = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_dt = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None
    
    files_to_process = []
    for f in files:
        # features_20200101.parquet
        try:
            dstr = f.stem.split("_")[1]
            dt = datetime.strptime(dstr, "%Y%m%d")
            
            if start_dt and dt < start_dt:
                continue
            if end_dt and dt > end_dt:
                continue
                
            files_to_process.append(f)
        except:
            continue
            
    print(f"Processing {len(files_to_process)} files after filtering.")
    
    for f in files_to_process:
        try:
            process_day(f)
        except Exception as e:
            print(f"❌ Error processing {f.name}: {e}")

if __name__ == "__main__":
    main()
