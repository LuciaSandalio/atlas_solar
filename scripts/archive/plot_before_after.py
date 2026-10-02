import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
    
    print("Cargando dataset calibrado...")
    df = pd.read_parquet('data/processed/stations/davis_calibrated_hourly_final.parquet')
    
    # Convert index to localized time for the plot
    df.index = pd.to_datetime(df.index).tz_convert('America/Argentina/Cordoba')
    
    df_vv = df[df['estacion'] == '531_villa_valeria_bioagroindustriacba'].copy()
    df_vv = df_vv.sort_index()
    
    # Split into pre and post
    date_split = pd.to_datetime('2021-01-01').tz_localize('America/Argentina/Cordoba')
    
    pre_2021_raw = df_vv.loc[:date_split, 'ghi_original'].dropna()
    post_2021_raw = df_vv.loc[date_split:, 'ghi_original'].dropna()
    
    pre_2021_cal = df_vv.loc[:date_split, 'solar_radiation'].dropna()
    post_2021_cal = df_vv.loc[date_split:, 'solar_radiation'].dropna()
    
    # Calculate P99.9 (excluding zeros/nights for a better peak representation)
    def calc_p99(series):
        active = series[series > 50]
        return np.percentile(active, 99.9) if len(active) > 0 else 0
        
    p99_pre_raw = calc_p99(pre_2021_raw)
    p99_post_raw = calc_p99(post_2021_raw)
    
    p99_pre_cal = calc_p99(pre_2021_cal)
    p99_post_cal = calc_p99(post_2021_cal)
    
    print(f"RAW: Pre P99={p99_pre_raw:.1f}, Post P99={p99_post_raw:.1f}")
    print(f"CALIBRATED: Pre P99={p99_pre_cal:.1f}, Post P99={p99_post_cal:.1f}")
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, sharey=True)
    
    # ---------------------------------------------------------
    # Panel 1: Original (RAW)
    # ---------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(df_vv.index, df_vv['ghi_original'], color='#3B82F6', linewidth=0.5, alpha=0.9)
    
    ax1.hlines(p99_pre_raw, df_vv.index.min(), date_split, colors='red', linestyles='dashed', 
               label=f'P99.9 pre-hueco ({p99_pre_raw:.1f} W/m²)')
    ax1.hlines(p99_post_raw, date_split, df_vv.index.max(), colors='orange', linestyles='dashed', 
               label=f'P99.9 post-hueco ({p99_post_raw:.1f} W/m²)')
               
    ax1.set_title('EMA: Villa Valeria - Serie Original (Sin Calibrar)', loc='left', weight='bold', fontsize=13)
    ax1.set_ylabel('Radiación Solar (W/m²)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')
    ax1.set_facecolor('#F8F9FA')
    ax1.set_ylim(0, 1500)
    
    # ---------------------------------------------------------
    # Panel 2: Calibrated (FINAL)
    # ---------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(df_vv.index, df_vv['solar_radiation'], color='#10B981', linewidth=0.5, alpha=0.9)
    
    ax2.hlines(p99_pre_cal, df_vv.index.min(), date_split, colors='red', linestyles='dashed', 
               label=f'P99.9 pre-hueco ({p99_pre_cal:.1f} W/m²)')
    ax2.hlines(p99_post_cal, date_split, df_vv.index.max(), colors='orange', linestyles='dashed', 
               label=f'P99.9 post-hueco ({p99_post_cal:.1f} W/m²)')
               
    ax2.set_title('EMA: Villa Valeria - Serie Calibrada Segmentada Anual (REST2)', loc='left', weight='bold', fontsize=13)
    ax2.set_ylabel('Radiación Solar (W/m²)')
    ax2.set_xlabel('Serie temporal horaria')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')
    ax2.set_facecolor('#F8F9FA')
    
    plt.tight_layout()
    
    paths = [
        Path("reports/figures/figura_villa_valeria_antes_despues.png"),
        Path("reportes/figures/figura_villa_valeria_antes_despues.png"),
        Path("/home/chula/.gemini/antigravity/brain/6c9b2b2f-9350-410d-8873-e5c094c49766/figura_villa_valeria_antes_despues.png")
    ]
    
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=250)
        print(f"Guardado en {p}")
        
    plt.close()

if __name__ == '__main__':
    main()
