import pandas as pd
import json

df = pd.read_parquet('data/processed/stations/davis_qc_hourly.parquet')
peaks = df.groupby(['estacion', df.index.hour])['solar_radiation'].mean().reset_index()
peak_hours = peaks.loc[peaks.groupby('estacion')['solar_radiation'].idxmax()].copy()

# Rename columns for clarity
peak_hours = peak_hours.rename(columns={'tiempo_utc': 'peak_hour'})

output = {
    "16_utc": peak_hours[peak_hours['peak_hour'] == 16]['estacion'].tolist(),
    "15_utc": peak_hours[peak_hours['peak_hour'] == 15]['estacion'].tolist(),
    "14_utc": peak_hours[peak_hours['peak_hour'] == 14]['estacion'].tolist(),
    "17_utc": peak_hours[peak_hours['peak_hour'] == 17]['estacion'].tolist()
}

with open("station_shifts.json", "w") as f:
    json.dump(output, f, indent=4)

print("Done")
