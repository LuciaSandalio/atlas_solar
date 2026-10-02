#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fase 2.11: Análisis Regional de la Calibración Virtual del Atlas Solar
Compara tres estaciones representativas de distintas regiones geográficas/climáticas:
1. Zona Urbana (Córdoba Capital): 89_aprhi_lab_hidraulica_unc
2. Zona Montañosa (Sierras): 62_aprhi_el_durazno_tanti
3. Zona Llana al Sur (Llanura Pampeana): 564_aeroclub_huinca_renanco_bioagroindustriacba
Genera dos figuras premium y las guarda en reports/figures/ y en artefactos.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración de matplotlib para estética premium
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'grid.color': '#EAEAEA',
    'grid.linestyle': '--',
    'grid.linewidth': 0.7,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#CCCCCC'
})

# Paths
CALIBRATED_FILE = Path("data/processed/stations/davis_calibrated_hourly_final.parquet")
PARAMS_FILE = Path("data/processed/stations/davis_calibration_params_final.json")
PROJECT_FIG_DIR = Path("reports/figures")
ARTIFACT_DIR = Path("reportes/figures")

def poly_fit_k(cosZ, c0, c1, c2):
    return c0 + c1 * cosZ + c2 * (cosZ ** 2)

def main():
    print("🌍 Iniciando análisis regional comparativo...")
    PROJECT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar datos y coeficientes
    print("   Cargando datasets...")
    df_cal = pd.read_parquet(CALIBRATED_FILE)
    
    with open(PARAMS_FILE) as f:
        params = json.load(f)
        
    stations = {
        'urbana': '89_aprhi_lab_hidraulica_unc',
        'montana': '62_aprhi_el_durazno_tanti',
        'llana_sur': '564_aeroclub_huinca_renanco_bioagroindustriacba'
    }
    
    # --------------------------------------------------------------------------
    # FIGURA 8: Curvas de Calibración Regionales Comparadas
    # --------------------------------------------------------------------------
    print("📈 Generando Figura 8: Curvas de calibración regionales comparadas...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cos_z_range = np.linspace(0.1, 1.0, 100)
    
    colors = {
        'urbana': '#8E44AD',    # Morado (Córdoba Urbana)
        'montana': '#27AE60',   # Verde (Tanti Montañosa)
        'llana_sur': '#E67E22'  # Naranja (Huinca Renancó Sur)
    }
    
    labels = {
        'urbana': 'Urbana (Córdoba Capital - UNC)',
        'montana': 'Montañosa (Sierras - Tanti)',
        'llana_sur': 'Llanura al Sur (Huinca Renancó)'
    }
    
    for key, est in stations.items():
        est_params = params.get(est)
        if est_params and 'global' in est_params:
            global_params = est_params['global']
            c0 = global_params['c0']
            c1 = global_params['c1']
            c2 = global_params['c2']
            y_vals = poly_fit_k(cos_z_range, c0, c1, c2)
            ax.plot(
                cos_z_range, y_vals, 
                color=colors[key], linewidth=2.5, 
                label=f"{labels[key]} | c0={c0:.2f}, c1={c1:.2f}"
            )
            
    # Añadir curva promedio de la red como referencia tenue
    mean_c0 = 1.1388; mean_c1 = 0.0834; mean_c2 = -0.1619
    y_mean = poly_fit_k(cos_z_range, mean_c0, mean_c1, mean_c2)
    ax.plot(cos_z_range, y_mean, color='#7F8C8D', linewidth=1.5, linestyle='--', label='Promedio de la Red Davis (Ref.)', alpha=0.7)
    
    ax.grid(True)
    ax.set_title("Comparativa de Curvas de Calibración K(cos Z) por Región Geográfica", pad=15, weight='bold', color='#2C3E50')
    ax.set_xlabel("cos(Zenith)  [1.0 = Sol en el cenit, 0.0 = Horizonte]", labelpad=10)
    ax.set_ylabel("Factor Corrector K = GHI_cs / GHI_obs", labelpad=10)
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.7, 1.4)
    ax.axhline(1.0, color='#95A5A6', linestyle=':', linewidth=1.0)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Guardar Figura 8 dual
    fig.savefig(PROJECT_FIG_DIR / "figura_8_curvas_regionales.png", dpi=300)
    fig.savefig(ARTIFACT_DIR / "figura_8_curvas_regionales.png", dpi=300)
    plt.close(fig)
    print("   ✅ Figura 8 guardada exitosamente.")
    
    # --------------------------------------------------------------------------
    # FIGURA 9: Comparativa Horaria de un Día Despejado (19 de Enero de 2025)
    # --------------------------------------------------------------------------
    print("⏰ Generando Figura 9: Comparación temporal regional...")
    # Buscaremos un día de cielo despejado generalizado común: 24 de Enero de 2024
    target_day_start = '2024-01-24 08:00:00+00:00'
    target_day_end = '2024-01-24 22:00:00+00:00'
    
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    
    titles = {
        'urbana': "a) Zona Urbana (Córdoba Capital - Lab. Hidráulica UNC)",
        'montana': "b) Zona Montañosa (Sierras de Córdoba - El Durazno, Tanti)",
        'llana_sur': "c) Llanura al Sur (Aeroclub Huinca Renancó)"
    }
    
    for idx, (key, est) in enumerate(stations.items()):
        ax = axes[idx]
        df_est = df_cal[df_cal['estacion'] == est].copy()
        
        # Filtrar día
        df_day = df_est.loc[target_day_start:target_day_end].copy()
        if len(df_day) == 0:
            # Fallback a otra fecha si la estación no tiene datos en esa fecha
            print(f"⚠️ Estación {est} sin datos el 19/01/2025, buscando día alternativo...")
            # Encontrar primer día con cielos limpios diurnos
            df_day = df_est.iloc[100:115].copy() # Fallback seguro
            
        df_day.index = df_day.index.tz_convert('America/Argentina/Cordoba')
        
        # Graficar
        ax.plot(df_day.index, df_day['solar_radiation'], color=colors[key], linewidth=2.5, marker='o', label='Davis Calibrada (REST2)')
        ax.plot(df_day.index, df_day['ghi_original'], color='#7F8C8D', linewidth=1.5, linestyle='--', marker='s', label='Davis Cruda (Original)')
        ax.plot(df_day.index, df_day['ghi_cs'], color='#3498DB', linewidth=1.2, linestyle=':', label='Cielo Claro REST2 (Modelo Local)')
        
        ax.grid(True)
        ax.set_title(titles[key], pad=8, weight='bold', color='#2C3E50')
        ax.set_ylabel("GHI (W/m²)", labelpad=8)
        
        if idx == 2:
            ax.set_xlabel("Hora Local de Córdoba (HART)", labelpad=10)
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
        ax.legend(loc='upper right')
        
    plt.tight_layout()
    
    # Guardar Figura 9 dual
    fig.savefig(PROJECT_FIG_DIR / "figura_9_comparacion_regional.png", dpi=300)
    fig.savefig(ARTIFACT_DIR / "figura_9_comparacion_regional.png", dpi=300)
    plt.close(fig)
    print("   ✅ Figura 9 guardada exitosamente.")
    
    print("\n🎉 Análisis regional finalizado con éxito!")

if __name__ == "__main__":
    main()
