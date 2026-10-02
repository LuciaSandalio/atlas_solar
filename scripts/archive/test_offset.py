import pandas as pd

df_feat = pd.read_parquet('data/ready/b0_model/features_with_B0.parquet')[['cell_id', 'time_round', 'cosZ', 'reflectance']]
df_davis = pd.read_parquet('data/processed/stations/davis_calibrated_hourly_final.parquet')
df_davis = df_davis.reset_index()

# Drop duplicates
stations = df_davis.drop_duplicates(subset=['estacion', 'latitude', 'longitude']).copy()

# A very rough mapping for speed
df_feat_unique = df_feat.drop_duplicates(subset=['cell_id']).copy()

print("Testing offsets...")
for offset in range(-5, 6):
    df_d = df_davis.copy()
    df_d['tiempo_utc'] = df_d['tiempo_utc'] + pd.Timedelta(hours=offset)
    
    # Just merge on time_round without cell_id, wait, we need cell_id
    # let's just shift df_merge that is already saved
    pass
