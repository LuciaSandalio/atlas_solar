import geopandas as gpd
#!/usr/import/env python3
# -*- coding: utf-8 -*-

"""
Fase 2.12: Regionalización Solar por Clustering de la Provincia de Córdoba
Adapta el pipeline de clustering meteorológico para agrupar las 166 estaciones Davis
en regiones de comportamiento solar homogéneo basadas en:
- Coordenadas (lat, lon)
- Altitud (altitude_m)
- Recurso Solar (mean_calibrated_ghi)
- Claridad del cielo (mean_clearness_index)
- Factor de Calibración Local (mean_calibration_factor)
Genera dos figuras premium y exporta un GeoPackage con los resultados.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

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
META_FILE = Path("data/raw/aws/resumen_estaciones_sol_todas.csv")
PROJECT_FIG_DIR = Path("reports/figures")
ARTIFACT_DIR = Path("/home/chula/.gemini/antigravity/brain/6c9b2b2f-9350-410d-8873-e5c094c49766")
OUTPUT_GPKG = Path("data/processed/stations/davis_solar_clusters.gpkg")

def main():
    print("🎨 Iniciando Regionalización Solar por Clustering (K-Means)...")
    PROJECT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar datos horarias
    print("   Cargando base de datos de radiación horaria...")
    df_cal = pd.read_parquet(CALIBRATED_FILE)
    
    # Filtrar solo horas diurnas para el análisis solar (GHI_cs > 10 W/m²)
    df_diurn = df_cal[df_cal['ghi_cs'] > 10.0].copy()
    df_diurn['clearness_index'] = (df_diurn['solar_radiation'] / df_diurn['ghi_cs']).clip(0.0, 1.2)
    
    # 2. Agregar métricas a nivel de estación
    print("   Calculando variables climáticas solares por estación...")
    df_est_stats = df_diurn.groupby('estacion').agg({
        'solar_radiation': 'mean',
        'clearness_index': 'mean',
        'calibration_factor': 'mean'
    }).rename(columns={
        'solar_radiation': 'mean_calibrated_ghi',
        'clearness_index': 'mean_clearness_index',
        'calibration_factor': 'mean_calibration_factor'
    }).reset_index()
    
    # 3. Cargar metadatos para altitud y filtrar estaciones aprobadas
    print("   Cruzando con altitud y coordenadas de metadatos...")
    meta_df = pd.read_csv(META_FILE)
    meta_df['lat_num'] = meta_df['lat'].astype(str).str.replace(',', '.').astype(float)
    meta_df['lon_num'] = meta_df['lon'].astype(str).str.replace(',', '.').astype(float)
    meta_df['queda_num'] = meta_df['queda'].astype(str).str.replace(',', '.').astype(float)
    
    # Filtrar solo las 166 estaciones aprobadas (queda en [1.0, 0.5, 0.7, 0.2])
    approved_meta = meta_df[meta_df['queda_num'].isin([1.0, 0.5, 0.7, 0.2])].copy()
    
    # Asegurar que altitude_m sea numérico
    approved_meta['altitude_m'] = pd.to_numeric(approved_meta['altitude_m'].astype(str).str.replace(',', '.'), errors='coerce')
    # Rellenar faltantes de altitud con la media regional si hubiera
    approved_meta['altitude_m'] = approved_meta['altitude_m'].fillna(approved_meta['altitude_m'].median())
    
    # Merge
    df_merge = df_est_stats.merge(
        approved_meta[['estacion', 'lat_num', 'lon_num', 'altitude_m', 'queda_num']],
        on='estacion',
        how='inner'
    )
    
    print(f"   ✓ Estaciones listas para clustering: {len(df_merge)}")
    
    # 4. Preparación de variables de clustering puramente físicas
    features = ['altitude_m', 'mean_calibrated_ghi', 'mean_clearness_index']
    print(f"   Variables seleccionadas: {features}")
    
    X = df_merge[features].copy()
    
    # Eliminar posibles filas con NaN
    X_clean = X.dropna()
    df_merge_clean = df_merge.loc[X_clean.index].copy()
    
    # Normalizar con StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # 5. Búsqueda del número óptimo de clusters (k)
    print("🔍 Evaluando número de clústeres óptimo (k)...")
    k_range = [3, 4, 5, 6]
    resultados = {}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels)
        resultados[k] = {'silhouette': silhouette, 'model': kmeans, 'labels': labels}
        print(f"   • k={k} | Silhouette Score = {silhouette:.3f}")
        
    # Seleccionar mejor k
    mejor_k = max(resultados.keys(), key=lambda k: resultados[k]['silhouette'])
    mejor_score = resultados[mejor_k]['silhouette']
    clusters_finales = resultados[mejor_k]['labels']
    print(f"🎯 El número óptimo de clústeres es k={mejor_k} (Silhouette: {mejor_score:.3f})")
    
    # Asignar clúster
    df_merge_clean['cluster'] = clusters_finales
    
    # Convertir a GeoDataFrame
    gdf_resultado = gpd.GeoDataFrame(
        df_merge_clean,
        geometry=gpd.points_from_xy(df_merge_clean['lon_num'], df_merge_clean['lat_num']),
        crs='EPSG:4326'
    )
    
    # --------------------------------------------------------------------------
    # FIGURA 10: Mapa Geográfico de Clusters Solares en Córdoba
    # --------------------------------------------------------------------------
    print("🗺️ Generando Figura 10: Mapa de regionalización solar...")
    
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
    
    # Colores elegantes para los clústeres (llaves int y str para robustez)
    colors_dict = {
        0: '#3498DB', 1: '#E67E22', 2: '#2ECC71', 3: '#8E44AD', 4: '#E74C3C',
        '0': '#3498DB', '1': '#E67E22', '2': '#2ECC71', '3': '#8E44AD', '4': '#E74C3C'
    }
    
    # Pilar
    ax.scatter(
        -63.8795, -31.6792,
        color='yellow',
        marker='*',
        s=350,
        edgecolors='black',
        linewidths=1.5,
        label='Referencia Pilar (CMP21)',
        zorder=5
    )
    
    for c in sorted(gdf_resultado['cluster'].unique()):
        cluster_gdf = gdf_resultado[gdf_resultado['cluster'] == c]
        ax.scatter(
            cluster_gdf['lon_num'], 
            cluster_gdf['lat_num'], 
            color=colors_dict.get(c, '#7F8C8D'),
            s=90,
            edgecolors='#2C3E50',
            linewidths=0.8,
            alpha=0.9,
            label=f'Región Solar {c}',
            zorder=3
        )
        
    ax.grid(True)
    ax.set_title("Regionalización Solar Climatológica de la Provincia de Córdoba\nClustering Multivariado K-Means (Red Davis)", pad=15, weight='bold', color='#2C3E50')
    
    # Ocultar ejes lat/lon según pedido
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Guardar Figura 10 dual
    fig.savefig(PROJECT_FIG_DIR / "figura_10_mapa_clusters_solares.png", dpi=300)
    fig.savefig(ARTIFACT_DIR / "figura_10_mapa_clusters_solares.png", dpi=300)
    plt.close(fig)
    print("   ✅ Figura 10 guardada.")
    
    # --------------------------------------------------------------------------
    # FIGURA 11: Diagnóstico Físico de Clústeres (Boxplots)
    # --------------------------------------------------------------------------
    print("📈 Generando Figura 11: Perfil de variables por clúster...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    vars_plot = [
        ('altitude_m', 'Altitud (msnm)', '#27AE60'),
        ('mean_calibrated_ghi', 'GHI Promedio Diurno (W/m²)', '#F39C12'),
        ('mean_clearness_index', 'Índice de Claridad (Kt)', '#E74C3C'),
        ('mean_calibration_factor', 'Factor de Calibración K', '#9B59B6')
    ]
    
    for idx, (col, title, color) in enumerate(vars_plot):
        ax = axes[idx]
        sns.boxplot(
            data=gdf_resultado,
            x='cluster',
            y=col,
            ax=ax,
            hue='cluster',
            palette=colors_dict,
            legend=False,
            width=0.6,
            linewidth=1.2
        )
        ax.grid(True)
        ax.set_title(title, weight='semibold', color='#2C3E50')
        ax.set_xlabel('Clúster Solar')
        ax.set_ylabel('')
        
    plt.suptitle("Caracterización Física de las Regiones Solares de Córdoba", weight='bold', y=0.98, color='#2C3E50', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Guardar Figura 11 dual
    fig.savefig(PROJECT_FIG_DIR / "figura_11_perfiles_clusters.png", dpi=300)
    fig.savefig(ARTIFACT_DIR / "figura_11_perfiles_clusters.png", dpi=300)
    plt.close(fig)
    print("   ✅ Figura 11 guardada.")
    
    # 6. Exportación a GeoPackage
    print(f"💾 Guardando GeoPackage regionalizado en {OUTPUT_GPKG}...")
    try:
        # Reproyectar a POSGAR 94 Argentina Faja 4
        gdf_export = gdf_resultado.to_crs(epsg=22174)
        gdf_export.to_file(OUTPUT_GPKG, driver='GPKG')
        print("   ✅ GeoPackage guardado correctamente.")
    except Exception as e:
        print(f"   ❌ Error al guardar GeoPackage: {e}")
        
    # 7. Resumen de Clústeres por Terminal
    print("\n" + "="*50)
    print("RESUMEN DE REGIONALIZACIÓN SOLAR")
    print("="*50)
    for c in sorted(gdf_resultado['cluster'].unique()):
        sub = gdf_resultado[gdf_resultado['cluster'] == c]
        print(f"\nRegión Solar {c} (n={len(sub)} estaciones):")
        print(f"   • Altitud Media: {sub['altitude_m'].mean():.1f} msnm")
        print(f"   • GHI Diurna Media: {sub['mean_calibrated_ghi'].mean():.1f} W/m²")
        print(f"   • Índice de Claridad: {sub['mean_clearness_index'].mean():.3f}")
        print(f"   • Factor de Calibración: {sub['mean_calibration_factor'].mean():.3f}")
    print("="*50 + "\n")
    
    # Guardar un resumen JSON en la carpeta de procesos
    resumen_dict = []
    for c in sorted(gdf_resultado['cluster'].unique()):
        sub = gdf_resultado[gdf_resultado['cluster'] == c]
        resumen_dict.append({
            "cluster": int(c),
            "cantidad_estaciones": int(len(sub)),
            "altitud_media_m": float(sub['altitude_m'].mean()),
            "ghi_media_w_m2": float(sub['mean_calibrated_ghi'].mean()),
            "indice_claridad_medio": float(sub['mean_clearness_index'].mean()),
            "factor_calibracion_medio": float(sub['mean_calibration_factor'].mean())
        })
    
    with open("data/processed/stations/davis_solar_clusters_summary.json", "w") as f:
        json.dump(resumen_dict, f, indent=4)
        
    print("🎉 Proceso de regionalización solar completado con éxito!")

if __name__ == "__main__":
    main()
