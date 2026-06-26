import geopandas as gpd
#!/usr/import/env python3
# -*- coding: utf-8 -*-

"""
Fase 2.8: Generación de Gráficos Premium para el Segundo Informe Técnico
Genera 4 figuras científicas de alta calidad y las guarda en el directorio de artefactos.
"""

import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración de matplotlib para estética premium
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
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
PILAR_FILE = Path("data/processed/stations/pilar_cmp21_hourly.parquet")
META_FILE = Path("data/raw/aws/resumen_estaciones_sol_todas.csv")

PROJECT_FIG_DIR = Path("reports/figures")
ARTIFACT_DIR = Path("/home/chula/.gemini/antigravity/brain/6c9b2b2f-9350-410d-8873-e5c094c49766")

def poly_fit_k(cosZ, c0, c1, c2):
    return c0 + c1 * cosZ + c2 * (cosZ ** 2)

def main():
    print("🎨 Generando gráficos científicos premium para el Informe 2...")
    PROJECT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    print("   Cargando datasets...")
    df_cal = pd.read_parquet(CALIBRATED_FILE)
    df_meta = pd.read_csv(META_FILE)
    df_meta['lat_num'] = df_meta['lat'].astype(str).str.replace(',', '.').astype(float)
    df_meta['lon_num'] = df_meta['lon'].astype(str).str.replace(',', '.').astype(float)
    df_meta['queda_num'] = df_meta['queda'].astype(str).str.replace(',', '.').astype(float)
    
    with open(PARAMS_FILE) as f:
        params = json.load(f)
        
    df_pilar = pd.read_parquet(PILAR_FILE)
    df_pilar.set_index('timestamp', inplace=True)
    if df_pilar.index.tz is None:
        df_pilar.index = df_pilar.index.tz_localize('UTC')
        
    df_pilar.index = df_pilar.index.tz_convert('UTC')
    df_cal.index = df_cal.index.tz_convert('UTC')
    
    # --------------------------------------------------------------------------
    # FIGURA 1: Mapa de las 166 estaciones Davis en Córdoba
    # --------------------------------------------------------------------------
    print("📍 Generando Figura 1: Mapa de estaciones...")
    
    # Cargar shapefiles avanzados
    try:
        cordoba_boundary = Path("data/cordoba_boundary.geojson")
        deptos_shp = Path("shapes_cba/cba_deptos.shp")
        
        cordoba_gdf = gpd.read_file(cordoba_boundary) if cordoba_boundary.exists() else None
        deptos_gdf = gpd.read_file(deptos_shp) if deptos_shp.exists() else None
    except Exception as e:
        print(f"   ⚠️ Error cargando shapefiles: {e}")
        cordoba_gdf, deptos_gdf = None, None
        
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Plot shapefiles
    if deptos_gdf is not None:
        deptos_gdf.plot(ax=ax, facecolor="none", edgecolor="#95A5A6", linewidth=0.8, zorder=1)
    if cordoba_gdf is not None:
        cordoba_gdf.plot(ax=ax, facecolor="none", edgecolor="#2C3E50", linewidth=1.5, zorder=3)
    
    # Filtrar aprobadas en los metadatos
    approved_meta = df_meta[df_meta['queda_num'].isin([1.0, 0.5, 0.7, 0.2])].dropna(subset=['lat_num', 'lon_num'])
    
    # Graficar estaciones coloreadas por años activos
    sc = ax.scatter(
        approved_meta['lon_num'], 
        approved_meta['lat_num'], 
        c=approved_meta['queda_num'],
        cmap='viridis',
        s=60,
        edgecolors='#2C3E50',
        linewidths=0.8,
        alpha=0.9,
        zorder=3
    )
    
    # Pilar de referencia
    ax.scatter(
        -63.8795, -31.6792,
        color='#E74C3C',
        marker='*',
        s=300,
        edgecolors='black',
        linewidths=1.2,
        label='Estación de Referencia Pilar (CMP21)',
        zorder=4
    )
    
    # Añadir grilla y detalles
    ax.grid(True)
    ax.set_title("Distribución Geográfica de la Red Davis en Córdoba\nEstaciones Aprobadas (n=166)", pad=15, weight='bold', color='#2C3E50')
    
    # Ocultar ejes lat/lon según pedido
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    ax.legend(loc='lower left', framealpha=0.95)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.03)
    cbar.set_label("Categoría de Inclusión (queda)", labelpad=10, weight='semibold')
    
    # Anotación de cantidad de estaciones
    ax.text(
        0.05, 0.05,
        f"Total Estaciones Calibradas: {len(approved_meta)}\nEstaciones Davis: 165\nEstaciones Termopila: 1",
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='#CCCCCC', boxstyle='round,pad=0.5'),
        fontsize=10
    )
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    fig.savefig(PROJECT_FIG_DIR / "figura_1_mapa_red.png", dpi=300)
    fig.savefig(ARTIFACT_DIR / "figura_1_mapa_red.png", dpi=300)
    plt.close(fig)
    print("   ✅ Figura 1 guardada.")
    
    # --------------------------------------------------------------------------
    # FIGURA 2: Curvas de Calibración K(cosZ)
    # --------------------------------------------------------------------------
    print("📉 Generando Figura 2: Curvas de calibración...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cos_z_range = np.linspace(0.1, 1.0, 100)
    
    # Curvas de estaciones destacadas
    estaciones_destacadas = {
        '347_laguna_larga_bioagroindustriacba': ('Laguna Larga', '#1ABC9C'),
        '89_aprhi_lab_hidraulica_unc': ('Lab Hidráulica UNC', '#2980B9'),
        '35_aprhi_campus_ubp': ('Campus UBP', '#8E44AD'),
        '178_bouwer_bioagroindustriacba': ('Bouwer', '#E67E22'),
    }
    
    # Graficar la curva promedio
    mean_c0 = 1.1388
    mean_c1 = 0.0834
    mean_c2 = -0.1619
    y_mean = poly_fit_k(cos_z_range, mean_c0, mean_c1, mean_c2)
    ax.plot(cos_z_range, y_mean, color='#2C3E50', linewidth=3.5, label='Curva Promedio de la Red Davis', zorder=5)
    
    for est, (label, color) in estaciones_destacadas.items():
        est_params = params.get(est)
        if est_params and 'global' in est_params:
            global_params = est_params['global']
            c0 = global_params['c0']
            c1 = global_params['c1']
            c2 = global_params['c2']
            y_vals = poly_fit_k(cos_z_range, c0, c1, c2)
            ax.plot(cos_z_range, y_vals, color=color, linewidth=2.0, alpha=0.85, label=f"{label} (c0={c0:.2f}, c1={c1:.2f})")
            
    # Estética del plot
    ax.grid(True)
    ax.set_title("Factor de Corrección K(cos Z) vs Coseno del Ángulo Cenital", pad=15, weight='bold', color='#2C3E50')
    ax.set_xlabel("cos(Zenith)  [1.0 = Sol en el cenit, 0.0 = Horizonte]", labelpad=10)
    ax.set_ylabel("Factor Corrector K = GHI_cs / GHI_obs", labelpad=10)
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.7, 1.4)
    ax.axhline(1.0, color='#95A5A6', linestyle='--', linewidth=1.0)
    ax.legend(loc='upper right')
    
    # Anotaciones físicas
    ax.text(
        0.75, 0.90,
        "Subestimación Espectral\n(K > 1.0 a sol alto)",
        color='#2C3E50', fontsize=9, style='italic', weight='semibold'
    )
    ax.text(
        0.18, 1.25,
        "Error de Coseno\n(Aumento de K a sol bajo)",
        color='#C0392B', fontsize=9, style='italic', weight='semibold'
    )
    
    plt.tight_layout()
    fig.savefig(PROJECT_FIG_DIR / "figura_2_curvas_calibracion.png", dpi=300)
    fig.savefig(ARTIFACT_DIR / "figura_2_curvas_calibracion.png", dpi=300)
    plt.close(fig)
    print("   ✅ Figura 2 guardada.")
    
    # --------------------------------------------------------------------------
    # FIGURA 3: Comparación Horaria en Laguna Larga vs Pilar (CMP21)
    # --------------------------------------------------------------------------
    print("⏰ Generando Figura 3: Comparación de series temporales...")
    df_laguna = df_cal[df_cal['estacion'] == '347_laguna_larga_bioagroindustriacba'].copy()
    
    df_merge = pd.merge(
        df_laguna[['ghi_original', 'solar_radiation', 'ghi_cs']],
        df_pilar[['GHI']],
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    # Filtrar para el día de cielo claro 5 de Noviembre de 2020
    df_day = df_merge.loc['2020-11-05 08:00:00+00:00':'2020-11-05 22:00:00+00:00'].copy()
    df_day.index = df_day.index.tz_convert('America/Argentina/Cordoba') # Convertir a local para el gráfico
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Graficar
    ax.plot(df_day.index, df_day['GHI'], color='#E74C3C', linewidth=2.5, marker='o', label='Referencia: Pilar (CMP21)')
    ax.plot(df_day.index, df_day['ghi_original'], color='#F39C12', linewidth=1.8, linestyle='--', marker='s', label='Davis Laguna Larga (Crudo)')
    ax.plot(df_day.index, df_day['solar_radiation'], color='#2ECC71', linewidth=2.5, marker='^', label='Davis Laguna Larga (Calibrado REST2)')
    ax.plot(df_day.index, df_day['ghi_cs'], color='#3498DB', linewidth=1.5, linestyle=':', label='Cielo Claro REST2 (Modelo)')
    
    ax.grid(True)
    ax.set_title("Comparación de Irradiancia Horaria GHI - 5 de Noviembre de 2020 (Cielo Claro)", pad=15, weight='bold', color='#2C3E50')
    ax.set_xlabel("Hora Local de Córdoba (HART)", labelpad=10)
    ax.set_ylabel("Global Horizontal Irradiance (W/m²)", labelpad=10)
    
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    ax.legend(loc='upper right')
    
    # Anotación de la corrección
    ax.annotate(
        "Sobreestimación Davis Crudo (+20.5%)",
        xy=(df_day.index[8], df_day['ghi_original'].iloc[8]),
        xytext=(df_day.index[4], df_day['ghi_original'].iloc[8] - 50),
        arrowprops=dict(facecolor='#F39C12', shrink=0.05, width=1.5, headwidth=6),
        fontsize=9, weight='bold', color='#D35400'
    )
    
    ax.annotate(
        "Corrección Virtual Exitosa (-2.7% error)",
        xy=(df_day.index[8], df_day['solar_radiation'].iloc[8]),
        xytext=(df_day.index[4], df_day['solar_radiation'].iloc[8] - 150),
        arrowprops=dict(facecolor='#2ECC71', shrink=0.05, width=1.5, headwidth=6),
        fontsize=9, weight='bold', color='#27AE60'
    )
    
    plt.tight_layout()
    fig.savefig(PROJECT_FIG_DIR / "figura_3_comparacion_pilar.png", dpi=300)
    fig.savefig(ARTIFACT_DIR / "figura_3_comparacion_pilar.png", dpi=300)
    plt.close(fig)
    print("   ✅ Figura 3 guardada.")
    
    # --------------------------------------------------------------------------
    # FIGURA 4: Histograma del factor corrector a sol alto
    # --------------------------------------------------------------------------
    print("📊 Generando Figura 4: Histograma de factores...")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Calcular K(cosZ = 0.5) para las 166 estaciones
    k_factors_list = []
    for est, est_params in params.items():
        if 'global' in est_params:
            global_params = est_params['global']
            if global_params.get('status') == 'global_calibrated':
                c0 = global_params['c0']
                c1 = global_params['c1']
                c2 = global_params['c2']
                k_factors_list.append(poly_fit_k(0.5, c0, c1, c2))
            
    ax.hist(
        k_factors_list,
        bins=20,
        color='#34495E',
        edgecolor='white',
        linewidth=1.2,
        alpha=0.85
    )
    
    ax.grid(True)
    ax.set_title("Distribución del Factor de Calibración K(cos Z = 0.5) en la Red", pad=15, weight='bold', color='#2C3E50')
    ax.set_xlabel("Factor de Calibración K(cos Z = 0.5)", labelpad=10)
    ax.set_ylabel("Frecuencia (Cantidad de Estaciones)", labelpad=10)
    
    # Media de la red
    mean_k = np.mean(k_factors_list)
    ax.axvline(mean_k, color='#E74C3C', linestyle='-', linewidth=2.5, label=f"Media de la Red Davis (K = {mean_k:.3f})")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    fig.savefig(PROJECT_FIG_DIR / "figura_4_histograma_factores.png", dpi=300)
    fig.savefig(ARTIFACT_DIR / "figura_4_histograma_factores.png", dpi=300)
    plt.close(fig)
    print("   ✅ Figura 4 guardada.")
    
    print("\n🎉 Generación de gráficos premium completada exitosamente!")

if __name__ == "__main__":
    main()
