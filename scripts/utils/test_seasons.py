import pandas as pd
import geopandas as gpd

print("Loading data...")
df_feat = pd.read_parquet('data/ready/b0_model/features_with_B0.parquet')
df_davis = pd.read_parquet('data/processed/stations/davis_calibrated_hourly_final.parquet')
if 'tiempo_utc' in df_davis.index.names: df_davis.reset_index(inplace=True)

df_ground = df_davis.copy()
df_ground.rename(columns={'tiempo_utc': 'timestamp', 'solar_radiation': 'target_GHI'}, inplace=True)

df_merge = pd.merge(df_ground, df_feat, left_on=['cell_id', 'timestamp'], right_on=['cell_id', 'time_round'], how='inner')

def get_season(month):
    if month in [12, 1, 2]: return 'Verano'
    elif month in [3, 4, 5]: return 'Otono'
    elif month in [6, 7, 8]: return 'Invierno'
    else: return 'Primavera'

df_merge['season'] = df_merge['timestamp'].dt.month.map(get_season)
print("Seasons found in df_merge:")
print(df_merge['season'].value_counts())
