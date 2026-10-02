import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import matplotlib.dates as mdates

def create_plots():
    print("Cargando predicciones de validación cruzada...")
    # Check if predictions exist
    pred_file = Path('results/fase4_cv_predictions.parquet')
    if not pred_file.exists():
        print("Error: No se encontró el archivo de predicciones.")
        return
        
    df = pd.read_parquet(pred_file)
    
    out_dir = Path("reportes/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14, 'axes.titlesize': 16, 'figure.titlesize': 20})
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- PLOT A: Scatter de Densidad (Hexbin) ---
    print("Generando Scatter de Densidad...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)
    
    # Take a random sample if dataset is too huge to plot quickly
    df_sample = df.sample(n=min(len(df), 500000), random_state=42)
    
    max_val = min(1200, df_sample['GHI_Real'].max() + 50)
    
    # Left: BD-JPT
    hb1 = ax1.hexbin(df_sample['GHI_Real'], df_sample['BD-JPT (Baseline)'], gridsize=50, cmap='YlOrRd', mincnt=1)
    ax1.plot([0, max_val], [0, max_val], 'k--', lw=2)
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.set_xlabel("GHI Observado (W/m²)", fontsize=12)
    ax1.set_ylabel("GHI Predicho (W/m²)", fontsize=12)
    ax1.set_title("Físico: BD-JPT (Baseline)", fontsize=14)
    
    # Right: Hybrid
    hb2 = ax2.hexbin(df_sample['GHI_Real'], df_sample['Híbrido (BD-JPT + ML)'], gridsize=50, cmap='YlGnBu', mincnt=1)
    ax2.plot([0, max_val], [0, max_val], 'k--', lw=2)
    ax2.set_xlabel("GHI Observado (W/m²)", fontsize=12)
    ax2.set_title("Machine Learning: Híbrido (BD-JPT + ML)", fontsize=14)
    
    fig.suptitle("Comparación de Dispersión (Validación Cruzada OOF)", fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(out_dir / "figura_fase4_scatter_density.png", dpi=120, bbox_inches='tight')
    plt.close()
    
    # --- PLOT B: Series Temporales por Región ---
    print("Generando Series Temporales por Región...")
    
    # Obtener una estación representativa por región
    regions = sorted(df['Region'].dropna().unique())
    stations_to_plot = {}
    for r in regions:
        # Buscamos la primera estación de esa región
        station = df[df['Region'] == r]['station_id'].iloc[0]
        stations_to_plot[r] = station
        
    for r, st_id in stations_to_plot.items():
        print(f"Generando Serie Temporal (Verano e Invierno) para Región {r} (Estación {st_id})...")
        
        # --- Verano ---
        df_verano = df[(df['station_id'] == st_id) & (df['Season'] == 'Verano')].copy()
        if not df_verano.empty:
            df_verano['timestamp'] = pd.to_datetime(df_verano['timestamp'])
            df_verano = df_verano.sort_values('timestamp')
            
            df_verano['date_only'] = df_verano['timestamp'].dt.floor('D')
            date_counts = df_verano.groupby('date_only').size()
            if not date_counts.empty:
                full_date_range = pd.date_range(start=date_counts.index.min(), end=date_counts.index.max(), freq='D')
                date_counts = date_counts.reindex(full_date_range, fill_value=0)
                rolling_7d = date_counts.rolling(window=7, min_periods=1).sum()
                best_end_date = rolling_7d.idxmax()
                best_start_date = best_end_date - pd.Timedelta(days=6)
                
                df_week = df_verano[(df_verano['timestamp'] >= best_start_date) & (df_verano['timestamp'] < best_end_date + pd.Timedelta(days=1))]
            
            if not df_week.empty:
                df_week = df_week.groupby('timestamp')[['GHI_Real', 'BD-JPT (Baseline)', 'Híbrido (BD-JPT + ML)']].mean()
                full_index = pd.date_range(start=df_week.index.min(), end=df_week.index.max(), freq='h')
                df_week = df_week.reindex(full_index).fillna(0)
                
                plt.figure(figsize=(14, 5))
                plt.plot(df_week.index, df_week['GHI_Real'], label='Observado', color='black', lw=2)
                plt.plot(df_week.index, df_week['BD-JPT (Baseline)'], label='BD-JPT (Físico)', color='#e74c3c', lw=1.5, alpha=0.8)
                plt.plot(df_week.index, df_week['Híbrido (BD-JPT + ML)'], label='Híbrido (ML)', color='#3498db', lw=1.5, alpha=0.9)
                
                plt.title(f"Reconstrucción Temporal de Irradiancia (Verano - Región {int(r)} - Estación {int(st_id)})", fontsize=18, fontweight='bold')
                plt.ylabel("GHI (W/m²)")
                plt.legend(loc='upper right')
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(out_dir / f"figura_fase4_timeseries_region{int(r)}_estacion{int(st_id)}_verano.png", dpi=120, bbox_inches='tight')
                plt.close()

        # --- Invierno ---
        df_invierno = df[(df['station_id'] == st_id) & (df['Season'] == 'Invierno')].copy()
        if not df_invierno.empty:
            df_invierno['timestamp'] = pd.to_datetime(df_invierno['timestamp'])
            df_invierno = df_invierno.sort_values('timestamp')
            
            df_invierno['date_only'] = df_invierno['timestamp'].dt.floor('D')
            date_counts = df_invierno.groupby('date_only').size()
            if not date_counts.empty:
                full_date_range = pd.date_range(start=date_counts.index.min(), end=date_counts.index.max(), freq='D')
                date_counts = date_counts.reindex(full_date_range, fill_value=0)
                rolling_7d = date_counts.rolling(window=7, min_periods=1).sum()
                best_end_date = rolling_7d.idxmax()
                best_start_date = best_end_date - pd.Timedelta(days=6)
                
                df_week = df_invierno[(df_invierno['timestamp'] >= best_start_date) & (df_invierno['timestamp'] < best_end_date + pd.Timedelta(days=1))]
            
            if not df_week.empty:
                df_week = df_week.groupby('timestamp')[['GHI_Real', 'BD-JPT (Baseline)', 'Híbrido (BD-JPT + ML)']].mean()
                full_index = pd.date_range(start=df_week.index.min(), end=df_week.index.max(), freq='h')
                df_week = df_week.reindex(full_index).fillna(0)
                
                plt.figure(figsize=(14, 5))
                plt.plot(df_week.index, df_week['GHI_Real'], label='Observado', color='black', lw=2)
                plt.plot(df_week.index, df_week['BD-JPT (Baseline)'], label='BD-JPT (Físico)', color='#e74c3c', lw=1.5, alpha=0.8)
                plt.plot(df_week.index, df_week['Híbrido (BD-JPT + ML)'], label='Híbrido (HGB)', color='#3498db', lw=1.5, alpha=0.9)
                
                plt.title(f"Reconstrucción Temporal de Irradiancia (Invierno - Región {int(r)} - Estación {int(st_id)})", fontsize=18, fontweight='bold')
                plt.ylabel("GHI (W/m²)")
                plt.legend(loc='upper right')
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(out_dir / f"figura_fase4_timeseries_region{int(r)}_estacion{int(st_id)}_invierno.png", dpi=120, bbox_inches='tight')
                plt.close()
    # --- PLOT C: Error vs Cenit ---
    print("Generando Perfil Cenital...")
    df['Zenith_Angle'] = np.rad2deg(np.arccos(df['cosZ']))
    
    bins = np.arange(0, 90, 5)
    df['Zenith_Bin'] = pd.cut(df['Zenith_Angle'], bins)
    
    df['Error_BDJPT'] = df['BD-JPT (Baseline)'] - df['GHI_Real']
    df['Error_Hibrido'] = df['Híbrido (BD-JPT + ML)'] - df['GHI_Real']
    
    mbe_bdjpt = df.groupby('Zenith_Bin', observed=False)['Error_BDJPT'].mean()
    mbe_hibrido = df.groupby('Zenith_Bin', observed=False)['Error_Hibrido'].mean()
    
    bin_centers = bins[:-1] + 2.5
    
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, mbe_bdjpt.values, marker='o', color='#e74c3c', label='BD-JPT (Baseline)')
    plt.plot(bin_centers, mbe_hibrido.values, marker='s', color='#3498db', label='Híbrido (BD-JPT + ML)')
    plt.axhline(0, color='black', linestyle='--', lw=1)
    
    plt.xlabel("Ángulo Cenital Solar (°)", fontsize=12)
    plt.ylabel("Sesgo Medio (MBE) [W/m²]", fontsize=12)
    plt.title("Dependencia Geométrica del Error Predictivo", fontsize=18, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "figura_fase4_error_zenith.png", dpi=120, bbox_inches='tight')
    plt.close()
    
    print("Gráficos avanzados generados en reportes/figures/")

if __name__ == "__main__":
    create_plots()
