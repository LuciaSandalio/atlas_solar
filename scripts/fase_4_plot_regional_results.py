import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Parse the LaTeX table to extract the final metrics for each Region-Season-Model
def extract_table():
    with open('results/bdjpt_regional_estacional_comparison.tex', 'r') as f:
        lines = f.readlines()
        
    data = []
    for line in lines:
        if ' & ' in line and not line.startswith('Región'):
            # Example line: R0 & Primavera & BD-JPT (Baseline) & -1.403 & -0.233 & 372.3 & 271.6 & -12.8 \\
            clean_line = line.replace('\\\\', '').replace('\n', '').strip()
            parts = [p.strip() for p in clean_line.split('&')]
            if len(parts) == 8:
                data.append(parts)
                
    df = pd.DataFrame(data, columns=['Region', 'Season', 'Model', 'R2_Kt', 'R2_GHI', 'RMSE', 'MAE', 'MBE'])
    
    # The table appended cumulative means for every fold (5 folds).
    # We only want the last entry for each Region-Season-Model combo
    df_final = df.drop_duplicates(subset=['Region', 'Season', 'Model'], keep='last').copy()
    
    # Convert numerical columns
    for col in ['R2_Kt', 'R2_GHI', 'RMSE', 'MAE', 'MBE']:
        df_final[col] = pd.to_numeric(df_final[col])
        
    return df_final

def plot_performance(df):
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Solo comparar BD-JPT vs RandomForest para simplificar
    df_plot = df[df['Model'].isin(['BD-JPT (Baseline)', 'RandomForest (Kt)'])]
    
    # Map season to correct order
    season_order = ['Verano', 'Otono', 'Invierno', 'Primavera']
    df_plot['Season'] = pd.Categorical(df_plot['Season'], categories=season_order, ordered=True)
    df_plot = df_plot.sort_values(['Region', 'Season'])

    regions = sorted(df_plot['Region'].unique())
    n_regions = len(regions)
    cols = 2
    rows = (n_regions + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), sharey=True)
    fig.suptitle('Desempeño de Calibración: RMSE (W/m²) por Región y Estación\nBaseline Físico vs Machine Learning', fontsize=18, fontweight='bold', y=0.95)
    
    # Flatten axes if necessary
    if rows == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
        
    for i, reg in enumerate(regions):
        ax = axes[i]
        df_reg = df_plot[df_plot['Region'] == reg]
        
        sns.barplot(data=df_reg, x='Season', y='RMSE', hue='Model', ax=ax, palette=['#e74c3c', '#2ecc71'])
        ax.set_title(f"Región {reg}", fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('RMSE (W/m²)' if i % cols == 0 else '')
        
        # Add values on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10)
            
        if i != 1:
            if ax.get_legend() is not None:
                ax.legend_.remove()
        else:
            if ax.get_legend() is not None:
                ax.legend(title='', fontsize=12)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('results/fase4_seasonal_rmse_comparison.png', dpi=300, bbox_inches='tight')
    print("Gráfico guardado en results/fase4_seasonal_rmse_comparison.png")

if __name__ == '__main__':
    df = extract_table()
    
    # Save a clean CSV table for the report
    clean_df = df.pivot_table(index=['Region', 'Season'], columns='Model', values='RMSE')
    clean_df.to_csv('results/fase4_clean_rmse_table.csv')
    print("Tabla limpia guardada en results/fase4_clean_rmse_table.csv")
    
    plot_performance(df)
