import geopandas as gpd
#!/usr/import/env python3
# -*- coding: utf-8 -*-

"""
Fase 2.9: Análisis Gráfico de Completitud de la Red Davis
Calcula las métricas de completitud y tasa de retención del QC para cada estación.
Genera dos figuras científicas premium y las guarda en el directorio de artefactos.
"""

import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Matplotlib styling for premium look
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
META_FILE = Path("data/raw/aws/resumen_estaciones_sol_todas.csv")
REPORT_OUT = Path("data/processed/stations/davis_completeness_metrics.csv")
PROJECT_FIG_DIR = Path("reports/figures")
ARTIFACT_DIR = Path("/home/chula/.gemini/antigravity/brain/6c9b2b2f-9350-410d-8873-e5c094c49766")

def main():
    print("📊 Iniciando análisis de completitud de datos y generación de gráficos...")
    PROJECT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar metadatos y dataset calibrado final
    print("   Cargando datasets...")
    meta_df = pd.read_csv(META_FILE)
    meta_df['lat_num'] = meta_df['lat'].astype(str).str.replace(',', '.').astype(float)
    meta_df['lon_num'] = meta_df['lon'].astype(str).str.replace(',', '.').astype(float)
    meta_df['queda_num'] = meta_df['queda'].astype(str).str.replace(',', '.').astype(float)
    
    # Filtrar solo las 166 estaciones aprobadas
    approved_meta = meta_df[meta_df['queda_num'].isin([1.0, 0.5, 0.7, 0.2])].copy()
    approved_stations = set(approved_meta['estacion'])
    
    df_cal = pd.read_parquet(CALIBRATED_FILE)
    
    # 2. Calcular métricas por estación
    print("   Calculando métricas de retención y completitud...")
    completeness_records = []
    
    for idx, row in approved_meta.iterrows():
        est = row['estacion']
        lat = row['lat_num']
        lon = row['lon_num']
        
        # Filtros de registros en el dataset calibrado final
        df_est = df_cal[df_cal['estacion'] == est]
        n_qc = len(df_est)
        
        # Registros originales estimación aproximada a partir de los metadatos o n_total
        n_total_meta = row['n_total']
        
        # Estimación de la tasa de retención (QC / Original)
        # La radiación horaria contiene 24 horas, pero el QC solo retiene diurnas (aprox 50%)
        # Por lo tanto, la tasa teórica máxima de retención es de aprox 50%
        # Calculamos la retención diurna respecto al total de horas diurnas originales esperadas
        retention_rate = (n_qc / (n_total_meta / 2.0)) * 100.0 if n_total_meta > 0 else 0.0
        retention_rate = min(retention_rate, 100.0) # Acotar a 100%
        
        # Años activos
        df_est_yearly = df_est.groupby(df_est.index.year)['solar_radiation'].count()
        n_years = len(df_est_yearly[df_est_yearly >= 200])
        
        completeness_records.append({
            "estacion": est,
            "lat": lat,
            "lon": lon,
            "n_original_total": n_total_meta,
            "n_qc_diurnal": n_qc,
            "retention_rate_pct": round(retention_rate, 2),
            "years_active": n_years,
            "queda": row['queda_num']
        })
        
    df_comp = pd.DataFrame(completeness_records)
    df_comp.to_csv(REPORT_OUT, index=False)
    print(f"   Métricas guardadas en {REPORT_OUT}")
    
    # --------------------------------------------------------------------------
    # FIGURA 5: Distribución de la Completitud y Tasa de Retención
    # --------------------------------------------------------------------------
    print("🎨 Generando Figura 5: Histograma y Boxplot de completitud...")
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Histograma de tasa de retención
    ax1.hist(
        df_comp['retention_rate_pct'],
        bins=15,
        color='#16A085',
        edgecolor='white',
        linewidth=1.2,
        alpha=0.85
    )
    ax1.grid(True)
    ax1.set_title("Distribución de la Fracción Seleccionada para Calibración", pad=12, weight='bold', color='#2C3E50')
    ax1.set_xlabel("Fracción de Cielo Claro Seleccionada (%)", labelpad=8)
    ax1.set_ylabel("Frecuencia (Cantidad de Estaciones)", labelpad=8)
    
    mean_ret = df_comp['retention_rate_pct'].mean()
    ax1.axvline(mean_ret, color='#E74C3C', linestyle='-', linewidth=2.5, label=f"Media: {mean_ret:.1f}%")
    ax1.legend(loc='upper left')
    
    plt.tight_layout()
    fig.savefig(PROJECT_FIG_DIR / "figura_5_completitud_distribucion.png", dpi=300)
    fig.savefig(ARTIFACT_DIR / "figura_5_completitud_distribucion.png", dpi=300)
    plt.close(fig)
    print("   ✅ Figura 5 guardada.")
    
    # --------------------------------------------------------------------------
    # FIGURA 6: Mapa Geográfico de Completitud en Córdoba
    # --------------------------------------------------------------------------
    print("🗺️ Generando Figura 6: Mapa geográfico de completitud...")
    
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
    
    # Graficar estaciones coloreadas por su tasa de retención
    sc = ax.scatter(
        df_comp['lon'], 
        df_comp['lat'], 
        c=df_comp['retention_rate_pct'],
        cmap='plasma',
        s=80,
        edgecolors='#2C3E50',
        linewidths=0.8,
        alpha=0.9,
        zorder=3
    )
    
    # Pilar
    ax.scatter(
        -63.8795, -31.6792,
        color='#2ECC71',
        marker='*',
        s=300,
        edgecolors='black',
        linewidths=1.2,
        label='Referencia Pilar (CMP21)',
        zorder=4
    )
    
    ax.grid(True)
    ax.set_title("Completitud Geográfica de la Red\nTasa de Retención Diurna por Estación", pad=15, weight='bold', color='#2C3E50')
    
    # Ocultar ejes lat/lon según pedido
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.03)
    cbar.set_label("Tasa de Retención Diurna (%)", labelpad=10, weight='semibold')
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    fig.savefig(PROJECT_FIG_DIR / "figura_6_mapa_completitud.png", dpi=300)
    fig.savefig(ARTIFACT_DIR / "figura_6_mapa_completitud.png", dpi=300)
    plt.close(fig)
    print("   ✅ Figura 6 guardada.")
    
    print("\n🎉 Generación de gráficos de completitud premium completada exitosamente!")

if __name__ == "__main__":
    main()
