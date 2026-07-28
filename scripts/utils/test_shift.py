import pandas as pd

df = pd.read_parquet('data/processed/stations/davis_calibrated_hourly_final.parquet')
df = df.reset_index()

audit = []
stations = df['estacion'].unique()

for s in stations[:10]:
    df_s = df[df['estacion'] == s].copy()
    envelope = df_s.groupby(df_s['tiempo_utc'].dt.hour)['solar_radiation'].quantile(0.90)
    peak_hour = envelope.idxmax()
    shift = 16 - peak_hour
    audit.append({'estacion': s, 'peak': peak_hour, 'shift': shift})

print(pd.DataFrame(audit))
