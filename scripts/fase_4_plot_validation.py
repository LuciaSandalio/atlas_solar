#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fase 4 (Auxiliar): Graficar resultados de la validación espacial BD-JPT.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score
from pathlib import Path
import os

# Robust RMSE import
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14, 'axes.titlesize': 18})
    
    val_file = Path("results/fase4_cv_predictions.parquet")
    if not val_file.exists():
        print(f"Error: No se encontró {val_file}")
        return
        
    df = pd.read_parquet(val_file)
    
    target = df['GHI_Real'].values
    pred = df['Híbrido (BD-JPT + ML)'].values
    
    # Calculate metrics
    rmse = root_mean_squared_error(target, pred)
    mbe = np.mean(pred - target)
    r2 = r2_score(target, pred)
    n = len(df)
    
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    
    # Scatter plot with hexbin or hist2d for density
    h = ax.hist2d(target, pred, bins=100, cmap='viridis', norm=LogNorm(), alpha=0.9)
    cb = fig.colorbar(h[3], ax=ax, label='Densidad de puntos (escala log)')
    
    # 1:1 line
    max_val = max(np.max(target), np.max(pred))
    ax.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=1.5, label='1:1 Ideal')
    
    # Styling
    ax.set_xlabel("GHI Observado (W/m²) - Red Davis")
    ax.set_ylabel("GHI Estimado Satelital (W/m²) - Modelo Híbrido")
    ax.set_title("Validación Espacial Independiente: Modelo Híbrido Final", fontsize=16, pad=15, weight='bold')
    
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Add text box with metrics
    textstr = '\n'.join((
        f'N = {n:,}',
        f'RMSE = {rmse:.1f} W/m²',
        f'MBE = {mbe:.1f} W/m²',
        f'$R^2$ = {r2:.3f}'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
            
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    out_paths = [
        Path("reportes/figuras/bdjpt_validation_scatter.png"),
        Path("results/bdjpt_validation_scatter.png")
    ]
    
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches='tight')
        print(f"Figura guardada en: {p}")
        
if __name__ == "__main__":
    main()
