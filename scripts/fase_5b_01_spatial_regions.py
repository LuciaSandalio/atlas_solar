#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 5B - Paso 01: Espacialización y Clasificación Regional a 2 km
==================================================================
1. Utiliza la grilla oficial inmutable de 2 km (data/grids/cordoba_grid_2km.geojson).
2. Reproyecta las 166 estaciones Davis terrestres y la grilla de 2 km a un CRS métrico
   oficial para Córdoba: POSGAR 2007 / Argentina 3 (EPSG:5345).
3. Entrena un clasificador 3-NN ponderado por el inverso de la distancia sobre coordenadas
   métricas planas (X, Y en metros).
4. Ejecuta validación cruzada espacial (Stratified 5-Fold y Leave-One-Out) sobre las 166 estaciones.
5. Calcula para cada celda de 2 km:
   - Región solar dominante (region: 0..3).
   - Pesos de consenso del vecindario (weight_R0 a weight_R{k-1}, k=3 regiones).
   - Consenso máximo del 3-NN (consenso_3nn = max(weights)).
   - Distancia euclidiana a la estación más cercana (dist_nn1_km).
   - Distancia media a las 3 estaciones más cercanas (dist_nn3_km).
6. Exporta el producto regional a data/ready/cordoba_grid_2km_regions.parquet y .geojson.
7. Genera figuras diagnósticas de consenso regional y matriz de confusión para el Informe 4.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'grid.color': '#DCDCDC',
    'grid.linestyle': '--',
    'grid.linewidth': 0.6
})

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
READY_DIR = DATA_DIR / "ready"
READY_DIR.mkdir(parents=True, exist_ok=True)
REPORTES_DIR = Path("reportes/figures")
REPORTES_DIR.mkdir(parents=True, exist_ok=True)

STATIONS_FILE = PROCESSED_DIR / "stations" / "davis_solar_clusters.gpkg"
BASE_GRID_2KM = DATA_DIR / "grids" / "cordoba_grid_2km.geojson"
OUT_GRID_PARQUET = READY_DIR / "cordoba_grid_2km_regions.parquet"
OUT_GRID_GEOJSON = READY_DIR / "cordoba_grid_2km_regions.geojson"

REGION_NAMES = {
    0: "Región 0 (Sierras Altas)",
    1: "Región 1 (Sierras Bajas / Transición)",
    2: "Región 2 (Llanura Pampeana)"
}
REGION_COLORS = {
    0: "#E67E22",  # Naranja
    1: "#27AE60",  # Verde
    2: "#2980B9"   # Azul
}

# CRS Métrico oficial de la Provincia de Córdoba: POSGAR 2007 Faja 3
METRIC_CRS = "EPSG:5345"

def main():
    print("=" * 80)
    print("🌍 FASE 5B-01: ESPACIALIZACIÓN REGIONAL 2 KM (CRS MÉTRICO POSGAR 2007 FAJA 3)")
    print("=" * 80)
    
    # 1. Cargar Estaciones y Reproyectar a CRS Métrico
    print(f"\n📂 1. Cargando estaciones desde {STATIONS_FILE}...")
    gdf_stations = gpd.read_file(STATIONS_FILE)
    n_stations = len(gdf_stations)
    print(f"   • Total estaciones Davis: {n_stations}")
    
    # Asegurar reproyección a POSGAR 2007 Faja 3
    if gdf_stations.crs != METRIC_CRS:
        gdf_stations_m = gdf_stations.to_crs(METRIC_CRS)
    else:
        gdf_stations_m = gdf_stations.copy()
        
    X_stations_m = np.column_stack([gdf_stations_m.geometry.x.values, gdf_stations_m.geometry.y.values])
    y_stations = gdf_stations_m['cluster'].values.astype(int)
    
    for c in sorted(np.unique(y_stations)):
        count_c = (y_stations == c).sum()
        pct_c = count_c / n_stations * 100.0
        print(f"     - {REGION_NAMES[c]}: {count_c} estaciones ({pct_c:.1f}%)")
        
    # 2. Validación Cruzada de Clasificadores Espaciales
    print(f"\n🔬 2. Evaluando clasificadores espaciales en coordenadas métricas planas...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    loo = LeaveOneOut()
    
    classifiers = {
        '1-NN (Métrico)': KNeighborsClassifier(n_neighbors=1),
        '3-NN (Métrico Ponderado)': KNeighborsClassifier(n_neighbors=3, weights='distance'),
        '5-NN (Métrico Ponderado)': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'RandomForest (Métrico)': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    }
    
    cv_summary = []
    for name, clf in classifiers.items():
        # Stratified 5-Fold
        preds_5f = cross_val_predict(clf, X_stations_m, y_stations, cv=skf)
        acc_5f = accuracy_score(y_stations, preds_5f)
        f1_5f = f1_score(y_stations, preds_5f, average='macro')
        
        # Leave-One-Out
        preds_loo = cross_val_predict(clf, X_stations_m, y_stations, cv=loo)
        acc_loo = accuracy_score(y_stations, preds_loo)
        f1_loo = f1_score(y_stations, preds_loo, average='macro')
        
        cv_summary.append({
            'Clasificador': name,
            'Acc (5F)': f"{acc_5f*100:.1f}\\%",
            'Macro-F1 (5F)': f"{f1_5f:.3f}",
            'Acc (LOO)': f"{acc_loo*100:.1f}\\%",
            'Macro-F1 (LOO)': f"{f1_loo:.3f}"
        })
        print(f"   • {name:<26} | Acc (5F): {acc_5f*100:.1f}% | Macro-F1: {f1_5f:.3f} | Acc (LOO): {acc_loo*100:.1f}%")
        
    df_cv = pd.DataFrame(cv_summary)
    
    # Exportar tabla LaTeX
    tex_path = REPORTES_DIR / "tabla_fase5b_validacion_clasificadores_regionales.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htpb]\n\\centering\n\\small\n")
        f.write("\\caption{Validación cruzada espacial de clasificadores territoriales (CRS métrico POSGAR 2007 Faja 3).}\n")
        f.write("\\label{tab:clasificadores_regionales_espaciales}\n")
        f.write(df_cv.to_latex(index=False, escape=False))
        f.write("\\end{table}")
    print(f"   ✓ Tabla LaTeX exportada: {tex_path}")
    
    # Modelo operativo: 3-NN Métrico Ponderado
    best_clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
    best_clf.fit(X_stations_m, y_stations)
    
    # Matriz de confusión en LOO
    preds_loo_best = cross_val_predict(best_clf, X_stations_m, y_stations, cv=loo)
    cm = confusion_matrix(y_stations, preds_loo_best)
    
    k_clusters = len(np.unique(y_stations))
    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    im = ax_cm.imshow(cm, cmap="Blues", interpolation="nearest")
    ax_cm.set_xticks(range(k_clusters))
    ax_cm.set_yticks(range(k_clusters))
    ax_cm.set_xticklabels([f"R{i}" for i in range(k_clusters)], weight='bold')
    ax_cm.set_yticklabels([f"R{i}" for i in range(k_clusters)], weight='bold')
    ax_cm.set_xlabel("Región Predicha (LOO)", weight='bold')
    ax_cm.set_ylabel("Cluster Solar Observado", weight='bold')
    ax_cm.set_title("Matriz de Confusión Espacial (3-NN Métrico, LOO)", weight='bold', pad=10)
    for i in range(k_clusters):
        for j in range(k_clusters):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max()/2 else "black", weight='bold')
    fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    fig_cm.savefig(REPORTES_DIR / "figura_fase5b_matriz_confusion_espacial.png", dpi=250)
    plt.close(fig_cm)
    print(f"   ✓ Matriz de confusión guardada: {REPORTES_DIR / 'figura_fase5b_matriz_confusion_espacial.png'}")
    
    # 3. Aplicar sobre la Grilla Oficial de 2 km (Inmutable en data/grids)
    print(f"\n🗺️ 3. Asignando pertenencia regional y pesos de consenso a la Grilla de 2 km...")
    gdf_grid_2km = gpd.read_file(BASE_GRID_2KM)
    n_2km = len(gdf_grid_2km)
    print(f"   • Celdas en grilla base: {n_2km:,}")
    
    # Reproyectar celdas a POSGAR 2007 Faja 3
    if gdf_grid_2km.crs != METRIC_CRS:
        gdf_grid_2km_m = gdf_grid_2km.to_crs(METRIC_CRS)
    else:
        gdf_grid_2km_m = gdf_grid_2km.copy()
        
    X_grid_m = np.column_stack([gdf_grid_2km_m.geometry.x.values, gdf_grid_2km_m.geometry.y.values])
    
    # Inferencia de región y pesos kNN
    region_preds = best_clf.predict(X_grid_m)
    weights_knn = best_clf.predict_proba(X_grid_m) # shape (59837, k_clusters)
    max_consensus = weights_knn.max(axis=1)
    
    # Cálculo de soporte geográfico (distancia a estaciones)
    nn_engine = NearestNeighbors(n_neighbors=3, metric='euclidean').fit(X_stations_m)
    distances_m, _ = nn_engine.kneighbors(X_grid_m)
    dist_nn1_km = np.round(distances_m[:, 0] / 1000.0, 2)
    dist_nn3_km = np.round(distances_m.mean(axis=1) / 1000.0, 2)
    
    print(f"   • Distancia media a estación más cercana: {dist_nn1_km.mean():.1f} km (máx: {dist_nn1_km.max():.1f} km)")
    print(f"   • Consenso regional medio (3-NN):         {max_consensus.mean():.3f}")
    
    # Armar GeoDataFrame enriquecido
    gdf_out = gdf_grid_2km.copy()
    gdf_out['region'] = region_preds.astype(int)
    gdf_out['region_name'] = [REGION_NAMES[r] for r in region_preds]
    for r in range(k_clusters):
        gdf_out[f'weight_R{r}'] = np.round(weights_knn[:, r], 4)
    gdf_out['consenso_3nn'] = np.round(max_consensus, 4)
    gdf_out['dist_nn1_km'] = dist_nn1_km
    gdf_out['dist_nn3_km'] = dist_nn3_km
    
    # Guardar productos (base original inmutable)
    gdf_out.to_parquet(OUT_GRID_PARQUET, index=False)
    print(f"   ✓ Grilla 2 km enriquecida guardada (Parquet): {OUT_GRID_PARQUET}")
    
    gdf_out.to_file(OUT_GRID_GEOJSON, driver="GeoJSON")
    print(f"   ✓ Grilla 2 km enriquecida guardada (GeoJSON): {OUT_GRID_GEOJSON}")
    
    # 4. Generar Figura Diagnóstica de Alta Resolución
    print(f"\n🎨 4. Generando mapa diagnóstico territorial...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5), constrained_layout=True)
    
    # Asegurar grados WGS84 para graficación cartográfica
    gdf_plot = gdf_out if gdf_out.crs == "EPSG:4326" else gdf_out.to_crs(4326)
    st_plot = gdf_stations if gdf_stations.crs == "EPSG:4326" else gdf_stations.to_crs(4326)
    
    # Panel Izquierdo: Regionalización Discreta
    ax0 = axes[0]
    for r in range(k_clusters):
        sub_g = gdf_plot[gdf_plot['region'] == r]
        ax0.scatter(sub_g.geometry.x, sub_g.geometry.y, s=1.2, color=REGION_COLORS[r],
                    label=REGION_NAMES[r], alpha=0.7)
                    
    # Estaciones terrestres
    ax0.scatter(st_plot.geometry.x, st_plot.geometry.y, c='black', s=22, edgecolors='white',
                linewidths=0.7, zorder=5, label=f"Estaciones Davis (n={n_stations})")
    ax0.set_title("Regionalización Territorial Discreta (3-NN en POSGAR 2007)\nAsignación de Modelo Dominante R0–R2", weight='bold', pad=10)
    ax0.set_xlabel("Longitud (°O)", weight='bold')
    ax0.set_ylabel("Latitud (°S)", weight='bold')
    ax0.grid(True)
    ax0.legend(loc='upper right', frameon=True, framealpha=0.92, markerscale=3)
    
    # Panel Derecho: Consenso y Soporte de Vecindad
    ax1 = axes[1]
    sc = ax1.scatter(gdf_plot.geometry.x, gdf_plot.geometry.y, c=gdf_plot['consenso_3nn'],
                     s=1.2, cmap='viridis', vmin=0.33, vmax=1.0, alpha=0.7)
    ax1.scatter(st_plot.geometry.x, st_plot.geometry.y, c='red', s=16, edgecolors='white',
                linewidths=0.5, zorder=5, label="Estaciones Davis")
    cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Consenso Regional del 3-NN ($w_{\\mathrm{max}}$)", weight='bold')
    ax1.set_title("Consenso Espacial de Vecindad (3-NN)\nZonas de Transición Suave (Candidatas a Spatial Blending)", weight='bold', pad=10)
    ax1.set_xlabel("Longitud (°O)", weight='bold')
    ax1.set_ylabel("Latitud (°S)", weight='bold')
    ax1.grid(True)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.92)
    
    fig_diag_path = REPORTES_DIR / "figura_fase5b_diagnostico_espacializacion_regiones.png"
    fig.savefig(fig_diag_path, dpi=250)
    plt.close(fig)
    print(f"   ✓ Mapa diagnóstico guardado: {fig_diag_path}")
    
    print("=" * 80)
    print("✅ FASE 5B-01 COMPLETADA CON ÉXITO: GRILLA 2 KM ENRIQUECIDA.")
    print("=" * 80)

if __name__ == "__main__":
    main()
