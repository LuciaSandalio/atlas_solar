
import xarray as xr
import sys

print("Testing xarray open_dataset...")
try:
    ds = xr.open_dataset('data/raw/atm/era5_atm_202001.nc', engine='h5netcdf')
    print("Success with engine='h5netcdf'")
    ds.close()
except Exception as e:
    print(f"Failed with h5netcdf: {e}")

try:
    ds = xr.open_dataset('data/raw/atm/era5_atm_202001.nc', engine='scipy')
    print("Success with engine='scipy'")
    ds.close()
except Exception as e:
    print(f"Failed with scipy: {e}")

try:
    ds = xr.open_dataset('data/raw/atm/era5_atm_202001.nc')
    print("Success with auto-detect")
    ds.close()
except Exception as e:
    print(f"Failed with auto-detect: {e}")
