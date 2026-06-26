#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cdsapi
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import calendar

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

# Bounds: North, West, South, East
# Extracted: -29.503, -65.778, -34.999, -61.778
# We add a buffer to ensure full coverage
AREA = [-29.0, -66.0, -35.5, -61.0] 

DATA_DIR = Path("data/raw/atm")


def get_era5_month(c, year, month, area, output_path):
    """
    Download ERA5 single levels for a specific month.
    Variables:
      - 'total_column_ozone' (tco3)
      - 'total_column_water_vapour' (tcwv)
      - 'surface_pressure' (sp)
      - 'total_aerosol_optical_depth_550nm' (from output of CAMS or CAMS reanalysis, 
        but ERA5 also has some aerosol. Use CAMS if possible. 
        For now, let's check if ERA5 has AOD available in 'reanalysis-era5-single-levels'.
        Actually, ERA5 does NOT have AOD in standard single levels. AOD is in CAMS.
        
        We will strictly download ERA5 core vars here. AOD needs a separate call if CAMS.
    """
    
    # ERA5 Request
    request = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'total_column_ozone', 
            'total_column_water_vapour', 
            'surface_pressure',
        ],
        'year': str(year),
        'month': f"{month:02d}",
        'day': [f"{d:02d}" for d in range(1, 32)],
        'time': [f"{h:02d}:00" for h in range(24)],
        'area': area,
    }

    print(f"   ⬇️  Requesting ERA5 {year}-{month:02d} -> {output_path}")
    c.retrieve('reanalysis-era5-single-levels', request, output_path)

def get_cams_month(c, year, month, area, output_path):
    """
    Download CAMS EAC4 AOD.
    Dataset: 'cams-global-reanalysis-eac4'
    Variable: 'total_aerosol_optical_depth_550nm'
    Time: 3-hourly (00, 03, ... 21).
    """
    request = {
        'format': 'netcdf',
        'variable': 'total_aerosol_optical_depth_550nm',
        'date': f"{year}-{month:02d}-01/{year}-{month:02d}-{calendar.monthrange(year, month)[1]}",
        'time': [f"{h:02d}:00" for h in range(0, 24, 3)],
        'area': area,
    }
    print(f"   ⬇️  Requesting CAMS {year}-{month:02d} -> {output_path}")
    c.retrieve('cams-global-reanalysis-eac4', request, output_path)


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 Atmospheric Data.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2020-01-01") # Default 1 day/month check
    parser.add_argument("--outdir", default=str(DATA_DIR))
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")

    # CDS API Client (ERA5)
    URL_CDS = "https://cds.climate.copernicus.eu/api"
    KEY_CDS = "044c553e-6be4-4238-a06c-61887c974d7e"
    c_era5 = cdsapi.Client(url=URL_CDS, key=KEY_CDS)

    # ADS API Client (CAMS)
    URL_ADS = "https://ads.atmosphere.copernicus.eu/api"
    KEY_ADS = "044c553e-6be4-4238-a06c-61887c974d7e" # Same key, different URL
    c_cams = cdsapi.Client(url=URL_ADS, key=KEY_ADS)

    # Iterate Months
    cur = start_date
    while cur <= end_date:
        year = cur.year
        month = cur.month
        
        # Output filename
        fname = f"era5_atm_{year}{month:02d}.nc"
        fpath = out_dir / fname
        
        if fpath.exists():
            print(f"   ✅ Exists: {fpath}, skipping.")
        else:
            try:
                get_era5_month(c_era5, year, month, AREA, str(fpath))
                print(f"   ✅ Downloaded: {fpath}")
            except Exception as e:
                print(f"   ❌ Failed ERA5 {year}-{month:02d}: {e}")

        # CAMS
        fpath_cams = out_dir / f"cams_aod_{year}{month:02d}.nc"
        if fpath_cams.exists():
             print(f"   ✅ Exists: {fpath_cams}, skipping.")
        else:
            try:
                get_cams_month(c_cams, year, month, AREA, str(fpath_cams))
                print(f"   ✅ Downloaded: {fpath_cams}")
            except Exception as e:
                print(f"   ❌ Failed CAMS {year}-{month:02d}: {e}")
        
        # Advance to next month
        # Logic: set day to 28, add 4 days, then extraction
        next_month = cur.replace(day=28) + timedelta(days=4)
        cur = next_month.replace(day=1)

    print("🎉 ERA5 Download Loop Finished.")

if __name__ == "__main__":
    main()
