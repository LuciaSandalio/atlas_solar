#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fase 2.10: Generación de Gráfico de Disponibilidad Temporal (Gantt)
Grafica la línea de tiempo de datos activos para cada una de las 166 estaciones.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Matplotlib styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 8,
    'figure.titlesize': 15,
    'grid.color': '#F0F0F0',
    'grid.linestyle': '-',
    'grid.linewidth': 0.5
})

# Paths
PARQUET_FILE = Path("data/processed/stations/davis_calibrated_hourly_final.parquet")
MATRIX_FILE = Path("data/processed/stations/davis_yearly_availability.csv")
PROJECT_FIG_DIR = Path("reports/figures")
ARTIFACT_DIR = Path("reportes/figures")

def main():
    print("📊 Generando gráfico de línea de tiempo de disponibilidad de datos...")
    
    if not PARQUET_FILE.exists():
        print(f"❌ Error: No se encuentra {PARQUET_FILE}")
        return
        
    print("📈 Cargando dataset final de irradiancia calibrada...")
    df_parq = pd.read_parquet(PARQUET_FILE)
    df_parq.index = pd.to_datetime(df_parq.index)
    df_parq['anio'] = df_parq.index.year
    
    # Calcular horas diurnas por estación y año
    counts = df_parq.groupby(['estacion', 'anio']).size().unstack(fill_value=0)
    
    # Asegurar que todas las columnas de años objetivo (2019-2025) estén presentes
    target_years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    for y in target_years:
        if y not in counts.columns:
            counts[y] = 0
            
    # Reordenar y guardar a CSV
    counts = counts[target_years]
    counts.to_csv(MATRIX_FILE, index=True)
    print(f"✅ Matriz de disponibilidad anual actualizada y guardada en {MATRIX_FILE}")
    
    df = pd.read_csv(MATRIX_FILE)
    df.set_index('estacion', inplace=True)
    
    # Ordenar estaciones por la cantidad de años activos y por nombre
    active_years_count = (df >= 200).sum(axis=1)
    df_sorted = df.loc[active_years_count.sort_values(ascending=True).index]
    
    # Crear plot tipo Gantt
    fig, ax = plt.subplots(figsize=(12, 18))
    
    stations = df_sorted.index.tolist()
    y_pos = np.arange(len(stations))
    
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    
    # Dibujar barras por cada año activo
    for i, est in enumerate(stations):
        for yr in years:
            hours = df_sorted.loc[est, str(yr)]
            if hours >= 200:
                # Color según intensidad de datos (horas de QC)
                color = '#1B4F72' if hours > 3000 else ('#2E86C1' if hours > 1000 else '#AED6F1')
                ax.barh(i, 1.0, left=yr-0.5, height=0.6, color=color, edgecolor='none')
                
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s.split('_')[1] + ' (' + s.split('_')[0] + ')' for s in stations], fontsize=7)
    
    ax.set_title("Cronograma de Disponibilidad de Datos de Radiación Solar\nRed Davis Calibrada (2019 - 2025)", pad=15, weight='bold', color='#2C3E50')
    ax.set_xlabel("Año de Medición", labelpad=10)
    ax.set_ylabel("Estación Meteorológica Davis (ID)", labelpad=10)
    
    ax.set_xlim(2018.5, 2025.5)
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=10, weight='semibold')
    
    # Grilla vertical
    ax.xaxis.grid(True, color='#DDDDDD', linestyle='--', linewidth=0.8)
    ax.yaxis.grid(False)
    
    # Leyenda personalizada
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1B4F72', label='Completa (> 3000 hs diurnas QC)'),
        Patch(facecolor='#2E86C1', label='Suficiente (1000 - 3000 hs diurnas QC)'),
        Patch(facecolor='#AED6F1', label='Parcial (200 - 1000 hs diurnas QC)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    PROJECT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path_proj = PROJECT_FIG_DIR / "figura_7_cronograma_estaciones.png"
    fig.savefig(fig_path_proj, dpi=250)
    print(f"   ✅ Figura 7 guardada en el proyecto: {fig_path_proj}")
    
    fig_path_art = ARTIFACT_DIR / "figura_7_cronograma_estaciones.png"
    fig.savefig(fig_path_art, dpi=250)
    plt.close(fig)
    print(f"   ✅ Figura 7 guardada en artefactos: {fig_path_art}")

if __name__ == "__main__":
    main()
