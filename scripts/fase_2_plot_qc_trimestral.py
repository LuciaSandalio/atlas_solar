import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates

def main():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
    
    # Cargar datos finales
    df = pd.read_parquet('data/processed/stations/davis_calibrated_hourly_final.parquet')
    df.index = pd.to_datetime(df.index).tz_convert('America/Argentina/Cordoba')
    
    stations_to_plot = [
        ('35_aprhi_campus_ubp', 'Ejemplo A: Falla por Sobreestimación (Picos irreales en 2020-2021)'),
        ('319_justiniano_posse_bioagroindustriacba', 'Ejemplo B: Falla por Subestimación (Sensor tapado/roto en 2025)')
    ]
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
    
    for ax, (est, title) in zip(axes, stations_to_plot):
        df_est = df[df['estacion'] == est].copy()
        df_est = df_est.sort_index()
        
        # Plot RAW Data (ghi_original)
        ax.plot(df_est.index, df_est['ghi_original'], color='#9CA3AF', linewidth=0.8, alpha=0.8, label='Serie Original (Cruda)')
        
        # Plot CALIBRATED Data (solar_radiation)
        ax.plot(df_est.index, df_est['solar_radiation'], color='#10B981', linewidth=1.0, alpha=1.0, label='Serie Calibrada (QC Dinámico Aplicado)')
        
        # Resaltar la zona teórica REST2
        ax.plot(df_est.index, df_est['ghi_cs'], color='orange', linewidth=0.5, alpha=0.5, label='Modelo REST2 (Límite Teórico)')
        
        ax.set_title(f'Estación {est} - {title}', weight='bold', fontsize=12, loc='left')
        ax.set_ylabel('Radiación Solar (W/m²)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        ax.set_facecolor('#F8F9FA')
        
        # Format X-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
    plt.tight_layout()
    
    out_paths = [
        Path("reports/figures/figura_qc_trimestral_efecto.png"),
        Path("reportes/figures/figura_qc_trimestral_efecto.png"),
        Path("/home/chula/.gemini/antigravity/brain/6c9b2b2f-9350-410d-8873-e5c094c49766/figura_qc_trimestral_efecto.png")
    ]
    
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=250)
        print(f"Guardado en {p}")

if __name__ == '__main__':
    main()
