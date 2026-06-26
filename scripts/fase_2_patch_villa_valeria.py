import pandas as pd
import json
import numpy as np
from pathlib import Path

def poly_fit_k(cosZ, c0, c1, c2):
    return c0 + c1 * cosZ + c2 * (cosZ ** 2)

def main():
    parquet_file = "data/processed/stations/davis_calibrated_hourly.parquet"
    json_file = "data/processed/stations/davis_calibration_params.json"
    
    df = pd.read_parquet(parquet_file)
    with open(json_file, 'r') as f:
        params = json.load(f)
        
    est = "531_villa_valeria_bioagroindustriacba"
    
    # Get 2020 params
    c0 = params[est]['yearly']['2020']['c0']
    c1 = params[est]['yearly']['2020']['c1']
    c2 = params[est]['yearly']['2020']['c2']
    
    # Apply to 2019
    params[est]['yearly']['2019']['c0'] = c0
    params[est]['yearly']['2019']['c1'] = c1
    params[est]['yearly']['2019']['c2'] = c2
    params[est]['yearly']['2019']['status'] = "patched_from_2020"
    
    # Update DataFrame for 2019
    df_vv = df[df['estacion'] == est]
    idx_2019 = df_vv.index[df_vv.index.tz_convert('America/Argentina/Cordoba').year == 2019]
    
    cosZ_2019 = df.loc[idx_2019, 'cosZ'].values
    k_factors = poly_fit_k(cosZ_2019, c0, c1, c2)
    k_factors = np.clip(k_factors, 0.6, 1.8)
    
    # Original is stored in 'ghi_original'
    # Recalculate solar_radiation
    df.loc[idx_2019, 'solar_radiation'] = df.loc[idx_2019, 'ghi_original'] * k_factors
    df.loc[idx_2019, 'calibration_factor'] = k_factors
    
    # Save back
    df.to_parquet(parquet_file)
    with open(json_file, 'w') as f:
        json.dump(params, f, indent=2)
        
    print("Patch applied to Villa Valeria 2019 successfully.")

if __name__ == "__main__":
    main()
