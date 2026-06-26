#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generación de Figuras 12, 13 y 14 para el Reporte Técnico del Atlas Solar de Córdoba.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Matplotlib premium styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 15,
    'grid.color': '#EAEAEA',
    'grid.linestyle': '--',
    'grid.linewidth': 0.7,
    'legend.fontsize': 9,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#CCCCCC'
})

CALIBRATED_FILE = Path("data/processed/stations/davis_calibrated_hourly_final.parquet")
PROJECT_FIG_DIR = Path("reports/figures")
REPORTES_FIG_DIR = Path("reportes/figures")
ARTIFACT_DIR = Path("/home/chula/.gemini/antigravity/brain/6c9b2b2f-9350-410d-8873-e5c094c49766")

def main():
    print("🎨 Iniciando la generación de gráficos personalizados (Figuras 12, 13 y 14)...")
    PROJECT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTES_FIG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    df = pd.read_parquet(CALIBRATED_FILE)
    df.index = pd.to_datetime(df.index)
    
    # ==========================================================================
    # FIGURA 12: Perfil Diario Cielo Claro Villa Valeria
    # ==========================================================================
    print("📈 Generando Figura 12...")
    df_val = df[df['estacion'] == '531_villa_valeria_bioagroindustriacba'].copy()
    
    # Buscar un día de cielo claro ideal (alta radiación, sin fluctuaciones)
    # Agrupamos por día y buscamos un día en Oct/Nov/Dic 2020 con alta radiación y estable
    df_val_2020 = df_val.loc['2020-01-01':'2020-12-31']
    daily_groups = df_val_2020.groupby(df_val_2020.index.date)
    best_day = None
    max_ghi = 0
    
    for date, group in daily_groups:
        if len(group) >= 10:  # Al menos 10 horas diurnas
            # Filtrar días con nubes (ghi_original / ghi_cs debe ser muy estable)
            ratio = group['ghi_original'] / group['ghi_cs']
            valid_ratio = ratio[(group['ghi_cs'] > 100) & (group['ghi_original'] > 50)]
            if len(valid_ratio) >= 6:
                std_ratio = valid_ratio.std()
                mean_ratio = valid_ratio.mean()
                # Queremos un día muy estable y que muestre la subestimación original (mean_ratio < 0.9)
                if std_ratio < 0.03 and mean_ratio < 0.85 and group['ghi_cs'].max() > max_ghi:
                    max_ghi = group['ghi_cs'].max()
                    best_day = date
                    
    if best_day is None:
        # Fallback a un día específico si no se encuentra automáticamente
        best_day = pd.to_datetime("2020-11-20").date()
        
    df_day = df_val.loc[str(best_day)].copy()
    df_day.index = df_day.index.tz_convert('America/Argentina/Cordoba')
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(df_day.index, df_day['ghi_original'], color='#E67E22', linewidth=2, linestyle='--', marker='s', label='Davis Cruda (Sobreestimación)')
    ax.plot(df_day.index, df_day['solar_radiation'], color='#27AE60', linewidth=2.5, marker='^', label='Davis Calibrada REST2 (Segmentada)')
    ax.plot(df_day.index, df_day['ghi_cs'], color='#2980B9', linewidth=1.5, linestyle=':', label='Cielo Claro REST2 (Modelo)')
    
    ax.grid(True)
    ax.set_title(f"Villa Valeria: Perfil Horario en Día Despejado ({best_day.strftime('%d/%m/%Y')})", pad=15, weight='bold', color='#2C3E50')
    ax.set_xlabel("Hora Local (HART)", labelpad=10)
    ax.set_ylabel("Global Horizontal Irradiance (W/m²)", labelpad=10)
    
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend(loc='upper right')
    
    # Anotaciones
    mid_idx = len(df_day) // 2
    ax.annotate(
        "Sobreestimación Severa (~20%)",
        xy=(df_day.index[mid_idx], df_day['ghi_original'].iloc[mid_idx]),
        xytext=(df_day.index[mid_idx] - pd.Timedelta(hours=3), df_day['ghi_original'].iloc[mid_idx] - 100),
        arrowprops=dict(facecolor='#E67E22', shrink=0.08, width=1, headwidth=5),
        fontsize=9, weight='bold', color='#D35400'
    )
    
    ax.annotate(
        "Corrección Dinámica REST2",
        xy=(df_day.index[mid_idx], df_day['solar_radiation'].iloc[mid_idx]),
        xytext=(df_day.index[mid_idx] - pd.Timedelta(hours=3), df_day['solar_radiation'].iloc[mid_idx] + 80),
        arrowprops=dict(facecolor='#27AE60', shrink=0.08, width=1, headwidth=5),
        fontsize=9, weight='bold', color='#27AE60'
    )
    
    plt.tight_layout()
    for d in [PROJECT_FIG_DIR, REPORTES_FIG_DIR, ARTIFACT_DIR]:
        fig.savefig(d / "figura_12_comparacion_villa_valeria.png", dpi=250)
    plt.close(fig)
    print("   ✅ Figura 12 guardada.")
    
    # ==========================================================================
    # FIGURA 13: Serie Temporal Completa Villa Valeria
    # ==========================================================================
    print("📈 Generando Figura 13...")
    # Resample mensual de horas diurnas (ghi_original > 5 W/m2)
    df_val_diurn = df_val[df_val['ghi_original'] > 5.0].copy()
    monthly_mean = df_val_diurn[['ghi_original', 'solar_radiation', 'ghi_cs']].resample('M').mean()
    
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(monthly_mean.index, monthly_mean['ghi_original'], color='#E67E22', linewidth=2, marker='o', alpha=0.8, label='Davis Villa Valeria (Cruda)')
    ax.plot(monthly_mean.index, monthly_mean['solar_radiation'], color='#27AE60', linewidth=2.5, marker='s', label='Davis Villa Valeria (Calibrada Segmentada)')
    ax.plot(monthly_mean.index, monthly_mean['ghi_cs'], color='#2980B9', linewidth=1.5, linestyle=':', label='Cielo Claro REST2 (Teórica)')
    
    ax.grid(True)
    ax.set_title("Evolución de la Serie Temporal Completa de Villa Valeria (2019-2025)\nImpacto Climatológico de la Calibración Segmentada Anual", pad=15, weight='bold', color='#2C3E50')
    ax.set_xlabel("Año", labelpad=10)
    ax.set_ylabel("GHI Diurna Media Mensual (W/m²)", labelpad=10)
    ax.legend(loc='lower left')
    
    # Destacar zona de subestimación
    ax.fill_between(monthly_mean.index, monthly_mean['ghi_original'], monthly_mean['solar_radiation'], color='#FFE6D5', alpha=0.5, label='Sesgo Corregido')
    
    plt.tight_layout()
    for d in [PROJECT_FIG_DIR, REPORTES_FIG_DIR, ARTIFACT_DIR]:
        fig.savefig(d / "figura_13_serie_completa_villa_valeria.png", dpi=250)
    plt.close(fig)
    print("   ✅ Figura 13 guardada.")
    
    # ==========================================================================
    # FIGURA 14: Comparativa Multi-Estación (4 estaciones representativas)
    # ==========================================================================
    print("📈 Generando Figura 14...")
    estaciones_multi = [
        ('531_villa_valeria_bioagroindustriacba', 'Villa Valeria (Sobreestimación)', '#E67E22', '#27AE60'),
        ('347_laguna_larga_bioagroindustriacba', 'Laguna Larga (Sobreestimación)', '#E74C3C', '#2ECC71'),
        ('89_aprhi_lab_hidraulica_unc', 'Lab Hidráulica UNC (Sesgo Espectral)', '#9B59B6', '#1ABC9C'),
        ('81_aprhi_em_ballesteros', 'EM Ballesteros (Desvío Estable)', '#F1C40F', '#34495E')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    
    for idx, (est_id, name, color_raw, color_cal) in enumerate(estaciones_multi):
        ax = axes[idx]
        df_est = df[df['estacion'] == est_id].copy()
        df_est_diurn = df_est[df_est['ghi_original'] > 5.0].copy()
        m_mean = df_est_diurn[['ghi_original', 'solar_radiation', 'ghi_cs']].resample('M').mean()
        
        ax.plot(m_mean.index, m_mean['ghi_original'], color=color_raw, linewidth=1.5, linestyle='--', marker='o', markersize=4, alpha=0.75, label='Original (Raw)')
        ax.plot(m_mean.index, m_mean['solar_radiation'], color=color_cal, linewidth=2, marker='s', markersize=4, label='Calibrada (Final)')
        ax.plot(m_mean.index, m_mean['ghi_cs'], color='#2980B9', linewidth=1, linestyle=':', label='Cielo Claro')
        
        ax.grid(True)
        ax.set_title(name, weight='bold', color='#2C3E50', fontsize=11)
        if idx % 2 == 0:
            ax.set_ylabel("GHI Diurna Media (W/m²)", fontsize=10)
        if idx >= 2:
            ax.set_xlabel("Año", fontsize=10)
        ax.legend(loc='lower left', fontsize=8)
        
    plt.suptitle("Comparativa de Series Temporales Completas (Antes vs Después de Calibración)\nEstaciones Clave Seleccionadas de la Red Davis (2019-2025)", y=0.96, weight='bold', color='#2C3E50', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    for d in [PROJECT_FIG_DIR, REPORTES_FIG_DIR, ARTIFACT_DIR]:
        fig.savefig(d / "figura_14_comparacion_multi_series.png", dpi=250)
    plt.close(fig)
    print("   ✅ Figura 14 guardada.")
    print("\n🎉 ¡Todos los gráficos personalizados de comparación han sido generados exitosamente!")

if __name__ == "__main__":
    main()
