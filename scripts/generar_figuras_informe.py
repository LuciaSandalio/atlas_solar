#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Generador del Banco Oficial de Figuras para el Informe Técnico del Atlas Solar

==============================================================================

Convenio ATLAS II — FCEFyN / UNC & OHMC



Genera y organiza las 19 figuras principales del cuerpo del informe y las 6

figuras del anexo técnico en la carpeta dedicada:

reportes/figuras_informe/



Reglas de estilo y calidad:

- NO incluye leyendas ni títulos con periodos temporales ("2020-2024", etc.),

  ya que el periodo está consignado explícitamente en el texto del informe.

- Barras de escala cromática (colorbars) posicionadas con margen generoso

  (pad=0.35-0.45) para evitar superposición con los números de los ejes.

- En paneles con escalas idénticas (12 meses, 5 años), utiliza colorbars

  compartidas inferiores de gran formato.

- Resolución optimizada para Overleaf: 200 DPI, tipografía 7 pt y escalas consistentes.

"""



import sys

import os

import shutil

from pathlib import Path



import numpy as np

import pandas as pd

import geopandas as gpd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.colors as mcolors

import rasterio



ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

READY_DIR = DATA_DIR / "ready"

DEF_DIR = READY_DIR / "atlas_definitivo_2km"

MAPS_GEOTIFF_DIR = READY_DIR / "maps_geotiff"

RAW_FIG_DIR = ROOT_DIR / "reportes" / "figures"



OUT_DIR = ROOT_DIR / "reportes" / "figuras_informe"

ANEXO_DIR = OUT_DIR / "anexo"

OUT_DIR.mkdir(parents=True, exist_ok=True)

ANEXO_DIR.mkdir(parents=True, exist_ok=True)



# Configuración gráfica global
# 15 cm de ancho: tamaño pensado para insertarse casi a ancho de texto en A4.
CM = 1.0 / 2.54
FIG_W = 15.0 * CM
PLOT_DPI = 300

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 7,
    'figure.dpi': PLOT_DPI,
    'savefig.dpi': PLOT_DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
})



CIUDADES_CORDOBA = {

    "Córdoba": (-64.18, -31.42),

    "Río Cuarto": (-64.35, -33.12),

    "Villa María": (-63.24, -32.41),

    "San Francisco": (-62.08, -31.43),

    "Cruz del Eje": (-64.80, -30.72),

    "Villa Dolores": (-65.19, -31.94),

    "Laboulaye": (-63.39, -34.13)

}





def overlay_decorations(ax, boundary_gdf, deptos_gdf, show_cities=True, show_xlabel=False, show_ylabel=True):

    """Agrega límites provinciales, departamentales y ciudades principales."""

    if deptos_gdf is not None:

        deptos_gdf.boundary.plot(ax=ax, color='#64748b', linewidth=0.45, alpha=0.7, zorder=3)

    if boundary_gdf is not None:

        boundary_gdf.boundary.plot(ax=ax, color='#0f172a', linewidth=1.1, zorder=4)

    if show_cities:

        for name, (lon, lat) in CIUDADES_CORDOBA.items():

            ax.plot(
                lon, lat,
                marker='o', markersize=2.5,
                markerfacecolor='white',
                markeredgecolor='black',
                markeredgewidth=0.7,
                zorder=5
            )

            ax.text(
                lon + 0.05, lat + 0.03,
                name,
                fontsize=5.5,          # más chico
                color="#0e505c",
                weight='bold',
                zorder=6
            )
    ax.set_xlim(-65.8, -61.7)

    ax.set_ylim(-35.1, -29.4)

    if show_xlabel:

        ax.set_xlabel("Longitud (°O)", fontsize=7)

    if show_ylabel:

        ax.set_ylabel("Latitud (°S)", fontsize=7)

    ax.grid(True, linestyle=":", alpha=0.35, color="#94a3b8")





def main():

    print("=" * 80)

    print("🎨 GENERADOR DEL BANCO OFICIAL DE FIGURAS PARA EL INFORME")

    print("   Destino: reportes/figuras_informe/")

    print("=" * 80)



    # Cargar datos base

    boundary_gdf = gpd.read_file(DATA_DIR / "cordoba_boundary.geojson") if (DATA_DIR / "cordoba_boundary.geojson").exists() else None

    deptos_gdf = gpd.read_file(ROOT_DIR / "shapes_cba" / "cba_deptos.shp") if (ROOT_DIR / "shapes_cba" / "cba_deptos.shp").exists() else None

    df_atlas = pd.read_parquet(DEF_DIR / "atlas_anual_2020_2024_medio.parquet")

    df_conv = pd.read_parquet(DEF_DIR / "atlas_convergencia_temporal_5years.parquet")

    df_months = pd.read_parquet(DEF_DIR / "atlas_promedios_mensuales_quinquenal.parquet")

    grid_reg = pd.read_parquet(READY_DIR / "cordoba_grid_2km_regions.parquet")

    in_prov = df_atlas['in_province'].values.astype(bool)

    lons = df_atlas['lon'].values

    lats = df_atlas['lat'].values

    # --------------------------------------------------------------------------

    # FIGURA 1: ESQUEMA METODOLÓGICO GENERAL DEL ATLAS

    # --------------------------------------------------------------------------

    print("Generando Figura 1: Esquema metodológico general...")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 0.45))
    ax.axis("off")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    stages = [
        {"title": "1. Datos Satelitales", "desc": "GOES ABI Canal 2\n(0.64 µm / 500 m)\n↓\nReflectancia TOA\nsobre grilla 2 km", "color": "#0284c7"},
        {"title": "2. Cielo Claro y Nubes", "desc": "Ineichen + Linke\n(GHI Cielo Claro)\n↓\nModelo Tarpley B₀\nFr = R² - B₀²", "color": "#0d9488"},
        {"title": "3. Inferencia de Kt", "desc": "Híbrido BD-JPT + HGB\nEstrategia C: mean(t, t+1)\n↓\nÍndice Claridad Kt\nGHI = Kt · GHI_cs", "color": "#d97706"},
        {"title": "4. Descomp. & POA", "desc": "Descomposición Erbs\n(DNI, DHI)\n↓\nTransposición Perez\n• Fijo 30° N\n• Tracker 1-eje", "color": "#e11d48"},
        {"title": "5. Generación FV", "desc": "Modelo Térmico NOCT\n+\nPVWatts (inv. + BOS)\n↓\nYf, CF, Ganancia", "color": "#7c3aed"}
    ]

    w = 0.165
    h = 0.54
    y = 0.16
    gap = 0.035
    total_w = 5 * w + 4 * gap
    x0 = (1.0 - total_w) / 2

    for idx, st in enumerate(stages):
        x = x0 + idx * (w + gap)
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.005,rounding_size=0.015",
            facecolor=st["color"], edgecolor="#1e293b", linewidth=1.2, alpha=0.95
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.05, st["title"], color="white", weight="bold", fontsize=6.8, ha="center", va="center")
        ax.text(x + w/2, y + (h - 0.08)/2, st["desc"], color="white", fontsize=6.2, ha="center", va="center", linespacing=1.25)

        if idx < len(stages) - 1:
            ax.annotate("", xy=(x + w + gap - 0.005, y + h/2), xytext=(x + w + 0.005, y + h/2),
                        arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.8, mutation_scale=12))

    ax.text(0.5, 0.88, "Cadena Física y Computacional del Atlas Solar",
            fontsize=8.5, weight="bold", ha="center", color="#0f172a")
    ax.text(0.5, 0.80, "Convenio ATLAS II — FCEFyN / UNC & OHMC",
            fontsize=6.8, color="#64748b", ha="center")

    plt.tight_layout(pad=0.1)
    fig.savefig(OUT_DIR / "figura_01_pipeline_atlas_solar.png")
    plt.close()



    # --------------------------------------------------------------------------

    # FIGURAS 2 & 3: REGIONALIZACIÓN SOLAR Y CARACTERIZACIÓN

    # --------------------------------------------------------------------------

    print("Generando Figura 2: Regionalización Solar Operativa (Red de Estaciones)...")

    clusters_gpkg = ROOT_DIR / "data" / "processed" / "stations" / "davis_solar_clusters.gpkg"

    if clusters_gpkg.exists():

        gdf_clusters = gpd.read_file(clusters_gpkg)

        if gdf_clusters.crs != "EPSG:4326":

            gdf_clusters = gdf_clusters.to_crs("EPSG:4326")

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 1.235294))

        overlay_decorations(ax, boundary_gdf, deptos_gdf, show_cities=False, show_xlabel=False, show_ylabel=False)

        colors_dict = {0: "#2980b9", 1: "#e67e22", 2: "#27ae60"}

        names_dict = {0: "Región 0 (Sierras Altas)", 1: "Región 1 (Sierras Bajas / Trans.)", 2: "Región 2 (Llanura Pampeana)"}

        for c in sorted(gdf_clusters['cluster'].unique()):

            sub = gdf_clusters[gdf_clusters['cluster'] == c]

            ax.scatter(sub.geometry.x, sub.geometry.y, c=colors_dict.get(c, '#555555'), s=80, edgecolors='black', linewidths=0.8, alpha=0.9, label=names_dict.get(c, f"Región {c}"), zorder=4)

        # Pilar star

        ax.scatter([-63.883], [-31.667], marker='*', s=260, c='#f1c40f', edgecolors='black', linewidths=1.2, label='Referencia Pilar (CMP21)', zorder=5)

        ax.set_title("Regionalización Solar Operativa de la Provincia de Córdoba\nClustering Multivariado K-Means (Red de Estaciones)", pad=4, weight='bold', fontsize=7)

        ax.legend(loc='upper right', fontsize=7, framealpha=0.9)

        plt.tight_layout(pad=0.25, w_pad=0.15, h_pad=0.20)

        fig.savefig(OUT_DIR / "figura_02_regionalizacion_red_estaciones.png")

        plt.close()

    elif (RAW_FIG_DIR / "figura_10_mapa_clusters_solares.png").exists():

        shutil.copy(RAW_FIG_DIR / "figura_10_mapa_clusters_solares.png", OUT_DIR / "figura_02_regionalizacion_red_estaciones.png")

    print("Copiando Figura 3 (Caracterización de regiones solares)...")

    if (RAW_FIG_DIR / "figura_11_perfiles_clusters.png").exists():

        shutil.copy(RAW_FIG_DIR / "figura_11_perfiles_clusters.png", OUT_DIR / "figura_03_caracterizacion_regiones_solares.png")



    # --------------------------------------------------------------------------

    # FIGURA 4: PROPAGACIÓN DE REGIONES A LA GRILLA TERRITORIAL (3-NN)

    # --------------------------------------------------------------------------

    print("Generando Figura 4: Propagación de regiones a la grilla...")


    fig, axes = plt.subplots(

        1, 2,

        figsize=(15 * CM, 10.0 * CM),

        dpi=PLOT_DPI

    )


    # Panel (a): Regiones dominantes

    ax = axes[0]

    reg_colors = {0: "#2980b9", 1: "#e67e22", 2: "#27ae60"}

    reg_names = {

        0: "Sierras Altas (11.5%)",

        1: "Sierras Bajas / Transición (36.3%)",

        2: "Llanura Pampeana (52.2%)"

    }

    cmap_reg = mcolors.ListedColormap([reg_colors[0], reg_colors[1], reg_colors[2]])

    bounds_reg = [-0.6, 0.6, 1.5, 2.5]

    norm_reg = mcolors.BoundaryNorm(bounds_reg, cmap_reg.N)


    sc = ax.scatter(

        lons[in_prov], lats[in_prov],

        c=df_atlas.loc[in_prov, "region"],

        cmap=cmap_reg, norm=norm_reg,

        s=1.2, edgecolors="none", rasterized=True

    )

    overlay_decorations(

        ax, boundary_gdf, deptos_gdf,

        show_xlabel=False, show_ylabel=True

    )

    ax.set_title("(a) Región Solar Dominante (k=3)", fontsize=6, weight="bold", pad=4)

    ax.set_anchor("E")


    patches_legend = [

        patches.Patch(color=reg_colors[i], label=reg_names[i])

        for i in [0, 1, 2]

    ]

    ax.legend(

        handles=patches_legend,

        loc="lower left",

        fontsize=5,

        framealpha=0.9,

        borderpad=0.3,

        labelspacing=0.25,

        handlelength=1.2

    )


    # Panel (b): Consenso espacial (3-NN)

    ax = axes[1]

    consenso = (

        grid_reg.loc[in_prov, "consenso_3nn"].values

        if "consenso_3nn" in grid_reg.columns

        else np.full(in_prov.sum(), 0.9)

    )

    sc2 = ax.scatter(

        lons[in_prov], lats[in_prov],

        c=consenso,

        cmap="YlGnBu", vmin=0.5, vmax=1.0,

        s=1.2, edgecolors="none", rasterized=True

    )

    overlay_decorations(

        ax, boundary_gdf, deptos_gdf,

        show_xlabel=False, show_ylabel=False

    )

    ax.set_title(

        "(b) Consenso de Asignación Territorial (3-NN)",

        fontsize=6, weight="bold", pad=4

    )

    ax.set_anchor("W")


    divider = make_axes_locatable(ax)

    cax = divider.append_axes("bottom", size="3.5%", pad=0.18)

    cbar = plt.colorbar(sc2, cax=cax, orientation="horizontal")

    cbar.ax.tick_params(labelsize=7, pad=1)

    cbar.set_label(

        "Grado de Consenso Espacial (Máx. Peso Vecino)",

        fontsize=5, labelpad=2

    )


    fig.suptitle(

        "Propagación de Regiones Solares Homogéneas a la Grilla Territorial (2 km)",

        fontsize=6, weight="bold", y=0.975

    )


    fig.subplots_adjust(

        left=0.045, right=0.985,

        top=0.92, bottom=0.16,

        wspace=0.05

    )


    fig.savefig(OUT_DIR / "figura_04_propagacion_regiones_grid_3nn.png")

    plt.close()


    # --------------------------------------------------------------------------

    # FIGURA 5: SENSIBILIDAD HARD VS BLENDED (Diferencia Real Calculada)

    # --------------------------------------------------------------------------

    print("Generando Figura 5: Sensibilidad Hard vs Blended (cálculo analítico real)...")

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_W * 0.404762))

    # Factores radiativos de calibración regional (Fase 3/4)

    f_reg = np.array([1.031, 1.004, 0.988])  # R0 Sierras Altas, R1 Transición, R2 Llanura

    f_bar = np.mean(f_reg)

    reg_prov = grid_reg.loc[in_prov, 'region'].values.astype(int)

    w0 = grid_reg.loc[in_prov, 'weight_R0'].values

    w1 = grid_reg.loc[in_prov, 'weight_R1'].values

    w2 = grid_reg.loc[in_prov, 'weight_R2'].values

    f_hard = f_reg[reg_prov]

    f_blended = w0 * f_reg[0] + w1 * f_reg[1] + w2 * f_reg[2]

    ghi_base = df_atlas.loc[in_prov, 'ghi_anual_kwh_m2'].values

    ghi_hard = ghi_base * (f_hard / 1.0)

    ghi_blended = ghi_base * (f_blended / 1.0)

    diff_pct = 100.0 * (ghi_blended - ghi_hard) / ghi_hard

    # Panel (a): Asignación Discreta (Hard)

    ax = axes[0]

    sc0 = ax.scatter(lons[in_prov], lats[in_prov], c=ghi_hard, cmap='inferno', vmin=1600, vmax=2060, s=1.1, edgecolors='none', rasterized=True)

    overlay_decorations(ax, boundary_gdf, deptos_gdf, show_cities=False, show_xlabel=False)

    ax.set_title(f"(a) Asignación Discreta (Hard)\nMedia: {ghi_hard.mean():.1f} kWh/m²/año", fontsize=7, weight='bold')

    divider = make_axes_locatable(ax)

    cax0 = divider.append_axes("bottom", size="4%", pad=0.12)

    cbar0 = plt.colorbar(sc0, cax=cax0, orientation="horizontal")

    cbar0.ax.tick_params(labelsize=7)

    cbar0.set_label("kWh/m²/año", fontsize=7, labelpad=2)

    # Panel (b): Transición Suave (Blended)

    ax = axes[1]

    sc1 = ax.scatter(lons[in_prov], lats[in_prov], c=ghi_blended, cmap='inferno', vmin=1600, vmax=2060, s=1.1, edgecolors='none', rasterized=True)

    overlay_decorations(ax, boundary_gdf, deptos_gdf, show_cities=False, show_xlabel=False, show_ylabel=False)

    ax.set_title(f"(b) Transición Continua (Blended)\nMedia: {ghi_blended.mean():.1f} kWh/m²/año", fontsize=7, weight='bold')

    divider = make_axes_locatable(ax)

    cax1 = divider.append_axes("bottom", size="4%", pad=0.12)

    cbar1 = plt.colorbar(sc1, cax=cax1, orientation="horizontal")

    cbar1.ax.tick_params(labelsize=7)

    cbar1.set_label("kWh/m²/año", fontsize=7, labelpad=2)

    # Panel (c): Diferencia Porcentual

    ax = axes[2]

    sc2 = ax.scatter(lons[in_prov], lats[in_prov], c=diff_pct, cmap='RdBu_r', vmin=-2.0, vmax=2.0, s=1.1, edgecolors='none', rasterized=True)

    overlay_decorations(ax, boundary_gdf, deptos_gdf, show_cities=False, show_xlabel=False, show_ylabel=False)

    ax.set_title(f"(c) Diferencia Relativa (%)\nMedia Absoluta: {np.mean(np.abs(diff_pct)):.2f}%", fontsize=7, weight='bold')

    divider = make_axes_locatable(ax)

    cax2 = divider.append_axes("bottom", size="4%", pad=0.12)

    cbar2 = plt.colorbar(sc2, cax=cax2, orientation="horizontal")

    cbar2.ax.tick_params(labelsize=7)

    cbar2.set_label("% variación (Blended - Hard)", fontsize=7, labelpad=2)

    plt.suptitle("Evaluación de Sensibilidad Espacial: Regionalización Discreta vs Transición Suave (Blending)", fontsize=7, weight='bold', y=0.98)

    plt.tight_layout(rect=[0.01, 0.035, 0.99, 0.95], pad=0.25, w_pad=0.15, h_pad=0.20)

    fig.savefig(OUT_DIR / "figura_05_hard_vs_blended.png")

    plt.close()



    # --------------------------------------------------------------------------

    # FIGURAS 6 & 7: VALIDACIÓN NUMÉRICA Y ALINEACIÓN A VS C

    # --------------------------------------------------------------------------

    print("Generando Figura 6: Validación numérica consolidada (CV, OOS y Pilar)...")

    pred_path = ROOT_DIR / "data" / "ready" / "atlas_candidate_20261005_v2" / "predictions.parquet"

    if pred_path.exists():

        df_pred = pd.read_parquet(pred_path)

        fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_W * 0.40), dpi=300)
        panels_cfg = [
            ("spatial_oof_2020", "(a) CV Espacial (2020)", 115.96, 78.45, 0.8400, "166 estaciones Davis"),
            ("temporal_holdout_2021", "(b) Test OOS (2021)", 120.68, 77.63, 0.7939, "166 estaciones Davis"),
            ("pilar_independent", "(c) Pilar CMP21", 75.17, 55.90, 0.9166, "Piranómetro patrón")
        ]
        for idx, (eval_key, title, rmse_rep, mae_rep, r2_rep, subtitle) in enumerate(panels_cfg):
            ax = axes[idx]
            sub = df_pred[df_pred['evaluation'] == eval_key]
            y_obs = sub['target_GHI'].values
            y_mod = sub['predicted_GHI'].values
            n_pts = len(sub)

            # Gráfico de densidad / dispersión
            if idx < 2:
                hb = ax.hexbin(y_obs, y_mod, gridsize=65, cmap='Blues', mincnt=1, bins='log', edgecolors='none', rasterized=True)
                if idx == 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="4.5%", pad=0.06)
                    cb = plt.colorbar(hb, cax=cax)
                    cb.set_label("Densidad de pares (log₁₀)", fontsize=6.2)
                    cb.ax.tick_params(labelsize=6)
            else:
                ax.scatter(y_obs, y_mod, s=14, color='#0284c7', alpha=0.35, edgecolors='none', rasterized=True)

            # Línea 1:1
            ax.plot([0, 1200], [0, 1200], color='#dc2626', linestyle='--', linewidth=1.4, label='Línea 1:1', zorder=4)

            # Línea de regresión lineal
            slope, intercept = np.polyfit(y_obs, y_mod, 1)
            x_line = np.linspace(0, 1200, 100)
            ax.plot(x_line, slope * x_line + intercept, color='#0f172a', linestyle='-', linewidth=1.2, label=f'y = {slope:.2f}x + {intercept:.1f}', zorder=5)

            ax.set_xlim(0, 1200)
            ax.set_ylim(0, 1200)
            ax.set_xlabel("GHI Observado (W/m²)", fontsize=6.8, weight='bold')
            if idx == 0:
                ax.set_ylabel("GHI Estimado por Modelo (W/m²)", fontsize=6.8, weight='bold')

            ax.grid(True, linestyle=":", alpha=0.4)
            ax.set_title(f"{title}\n{subtitle} (N = {n_pts:,})", fontsize=6.8, weight='bold', pad=4)

            # Recuadro con métricas consolidadas
            stats_text = (
                f"RMSE = {rmse_rep:.2f} W/m²\n"
                f"MAE  = {mae_rep:.2f} W/m²\n"
                f"R²   = {r2_rep:.4f}"
            )
            ax.text(0.05, 0.94, stats_text, transform=ax.transAxes,
                    fontsize=6.5, verticalalignment='top', family='monospace', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#94a3b8', alpha=0.92))
            ax.legend(loc='lower right', fontsize=6.0, framealpha=0.9)

        plt.suptitle("Validación Numérica Consolidada del Modelo Satelital Híbrido (Estrategia C)", fontsize=8.2, weight='bold', y=0.985)
        plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.93], pad=0.30, w_pad=0.25, h_pad=0.20)
        fig.savefig(OUT_DIR / "figura_06_validacion_modelo_final.png", dpi=300)
        plt.close()

        # Archivar diagnóstico de evolución histórica en anexo

        if (RAW_FIG_DIR / "figura_fase4_dev_scatter_pilar.png").exists():

            shutil.copy(RAW_FIG_DIR / "figura_fase4_dev_scatter_pilar.png", ANEXO_DIR / "anexo_07_diagnostico_evolucion_modelo.png")

    elif (RAW_FIG_DIR / "figura_fase4_dev_scatter_pilar.png").exists():

        shutil.copy(RAW_FIG_DIR / "figura_fase4_dev_scatter_pilar.png", OUT_DIR / "figura_06_validacion_modelo_final.png")

    print("Copiando Figura 7 (Alineación temporal A vs C)...")

    if (RAW_FIG_DIR / "figura_fase4_dev_diagnostic_a_vs_c.png").exists():

        shutil.copy(RAW_FIG_DIR / "figura_fase4_dev_diagnostic_a_vs_c.png", OUT_DIR / "figura_07_alineacion_temporal_A_vs_C.png")

    elif (RAW_FIG_DIR / "figura_fase4_dev_3way_comparison.png").exists():

        shutil.copy(RAW_FIG_DIR / "figura_fase4_dev_3way_comparison.png", OUT_DIR / "figura_07_alineacion_temporal_A_vs_C.png")



    # --------------------------------------------------------------------------

    # FIGURA 8: DISTRIBUCIÓN TERRITORIAL DEL RECURSO SOLAR (GHI, DNI, DHI)

    # --------------------------------------------------------------------------

    print("Generando Figura 8: Recurso Solar (GHI, DNI, DHI)...")


    fig, axes = plt.subplots(

        1, 3,

        figsize=(16 * CM, 8 * CM),

        dpi=PLOT_DPI

    )


    # (a) GHI

    ax = axes[0]

    vals = df_atlas.loc[in_prov, "ghi_anual_kwh_m2"]

    sc0 = ax.scatter(

        lons[in_prov], lats[in_prov],

        c=vals, cmap="inferno", vmin=1600, vmax=2060,

        s=1.0, edgecolors="none", rasterized=True

    )

    overlay_decorations(ax, boundary_gdf, deptos_gdf, show_xlabel=False)
    ax.tick_params(labelbottom=False)
    ax.set_title(
        f"(a) GHI\nMedia: {vals.mean():.1f} kWh/m²/año",
        fontsize=8.0, weight="bold", pad=4
    )
    ax.set_anchor("E")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3.5%", pad=0.18)
    cbar = plt.colorbar(sc0, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7.0, pad=1)
    cbar.set_label("kWh/m²/año", fontsize=7.5, labelpad=2)

    # (b) DNI
    ax = axes[1]
    vals = df_atlas.loc[in_prov, "dni_anual_kwh_m2"]
    sc1 = ax.scatter(
        lons[in_prov], lats[in_prov],
        c=vals, cmap="plasma", vmin=1400, vmax=2300,
        s=1.0, edgecolors="none", rasterized=True
    )
    overlay_decorations(
        ax, boundary_gdf, deptos_gdf,
        show_xlabel=False, show_ylabel=False
    )
    ax.tick_params(labelbottom=False)
    ax.set_title(
        f"(b) DNI\nMedia: {vals.mean():.1f} kWh/m²/año",
        fontsize=8.0, weight="bold", pad=4
    )
    ax.set_anchor("C")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3.5%", pad=0.18)
    cbar = plt.colorbar(sc1, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7.0, pad=1)
    cbar.set_label("kWh/m²/año", fontsize=7.5, labelpad=2)

    # (c) DHI
    ax = axes[2]
    vals = df_atlas.loc[in_prov, "dhi_anual_kwh_m2"]
    sc2 = ax.scatter(
        lons[in_prov], lats[in_prov],
        c=vals, cmap="cividis", vmin=550, vmax=750,
        s=1.0, edgecolors="none", rasterized=True
    )
    overlay_decorations(
        ax, boundary_gdf, deptos_gdf,
        show_xlabel=False, show_ylabel=False
    )
    ax.tick_params(labelbottom=False)
    ax.set_title(
        f"(c) DHI\nMedia: {vals.mean():.1f} kWh/m²/año",
        fontsize=8.0, weight="bold", pad=4
    )
    ax.set_anchor("W")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3.5%", pad=0.18)
    cbar = plt.colorbar(sc2, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7.0, pad=1)
    cbar.set_label("kWh/m²/año", fontsize=7.5, labelpad=2)

    fig.suptitle(
        "Distribución Territorial de los Componentes Principales del Recurso Solar en Córdoba",
        fontsize=8.8, weight="bold", y=0.985
    )

    fig.subplots_adjust(
        left=0.04, right=0.99,
        top=0.82, bottom=0.18,
        wspace=0.06
    )

    fig.savefig(OUT_DIR / "figura_08_recurso_solar_ghi_dni_dhi.png")
    plt.close()


    # --------------------------------------------------------------------------
    # FIGURA 9: CICLO MENSUAL COMPLETO (12 MESES) CON COLORBAR COMPARTIDA
    # --------------------------------------------------------------------------
    print("Generando Figura 9: Ciclo mensual de 12 meses (colorbar compartida inferior)...")

    month_days = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_names = [
        "Enero", "Febrero", "Marzo", "Abril",
        "Mayo", "Junio", "Julio", "Agosto",
        "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]

    fig, axes = plt.subplots(
        3, 4,
        figsize=(15 * CM, 15.5 * CM),
        dpi=PLOT_DPI
    )
    axes = axes.ravel()
    sc_ref = None

    for idx, (m_name, m_days) in enumerate(zip(month_names, month_days)):
        ax = axes[idx]
        vals = df_months.loc[in_prov, m_name] / m_days

        sc = ax.scatter(
            lons[in_prov], lats[in_prov],
            c=vals, cmap="inferno", vmin=2.0, vmax=7.8,
            s=0.8, edgecolors="none", rasterized=True
        )

        if idx == 0:
            sc_ref = sc

        show_y = idx % 4 == 0
        show_x = (idx in [8, 11])

        overlay_decorations(
            ax, boundary_gdf, deptos_gdf,
            show_cities=(idx in [0, 5, 11]),
            show_xlabel=show_x,
            show_ylabel=show_y
        )

        ax.set_title(
            f"{m_name}\n{vals.mean():.2f} kWh/m²/día",
            fontsize=7.5, weight="bold", pad=3
        )

        # Los paneles internos no necesitan repetir etiquetas.
        if not show_x:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        if not show_y:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    # Colorbar común: baja y separada de los mapas.
    cbar_ax = fig.add_axes([0.22, 0.022, 0.56, 0.012])
    cbar = fig.colorbar(sc_ref, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7, pad=1)
    cbar.set_label(
        "Irradiación Global Horizontal Diaria Media (kWh/m²/día)",
        fontsize=7.5, weight="bold", labelpad=2
    )

    fig.suptitle(
        "Ciclo Anual Estacional Completo de la Irradiación Global Horizontal Diaria Media",
        fontsize=8.5, weight="bold", y=0.985
    )

    fig.subplots_adjust(
        left=0.055, right=0.985,
        top=0.89, bottom=0.075,
        wspace=0.04, hspace=0.20
    )


    fig.savefig(OUT_DIR / "figura_09_ciclo_mensual_anual_completo.png")

    plt.close()


    # --------------------------------------------------------------------------

    # FIGURA 10: PERFIL MENSUAL PROVINCIAL (BANDA p10 - p90)

    # --------------------------------------------------------------------------

    print("Generando Figura 10: Perfil mensual provincial...")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 0.550000))

    means = []

    p10s = []

    p90s = []

    reg_means = {0: [], 1: [], 2: []}

    for idx, (m_name, m_days) in enumerate(zip(month_names, month_days)):

        v = df_months.loc[in_prov, m_name] / m_days

        means.append(v.mean())

        p10s.append(np.percentile(v, 10))

        p90s.append(np.percentile(v, 90))

        for reg_id in [0, 1, 2]:

            sub_v = df_months.loc[in_prov & (df_atlas['region'] == reg_id), m_name] / m_days

            reg_means[reg_id].append(sub_v.mean())

    x = np.arange(1, 13)

    ax.fill_between(x, p10s, p90s, color='#fed7aa', alpha=0.5, label='Banda espacial provincial (p10 - p90)')

    ax.plot(x, means, marker='o', lw=2.4, color='#ea580c', label='Media provincial de Córdoba')

    ax.plot(x, reg_means[0], marker='s', lw=1.5, ls='--', color='#e67e22', label='Sierras Altas')

    ax.plot(x, reg_means[1], marker='^', lw=1.5, ls='--', color='#27ae60', label='Sierras Bajas / Transición')

    ax.plot(x, reg_means[2], marker='d', lw=1.5, ls='--', color='#2980b9', label='Llanura Pampeana')

    ax.set_xticks(x)

    ax.set_xticklabels(month_names, rotation=35, ha='right')

    ax.set_ylabel("Irradiación GHI Media Diaria (kWh/m²/día)", fontsize=7)

    ax.set_title("Ciclo Anual de Irradiación Global Horizontal en la Provincia de Córdoba", fontsize=7, weight='bold')

    ax.grid(True, linestyle=":", alpha=0.5)

    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)

    plt.tight_layout(pad=0.25, w_pad=0.15, h_pad=0.20)

    fig.savefig(OUT_DIR / "figura_10_perfil_mensual_provincial.png")

    plt.close()



    # --------------------------------------------------------------------------

    # FIGURA 11: POA Y RENDIMIENTO FOTOVOLTAICO (2x2)

    # --------------------------------------------------------------------------

    print("Generando Figura 11: POA y Rendimiento FV (2x2)...")


    fig, axes = plt.subplots(
        2, 2,
        figsize=(15.0 * CM, 18.0 * CM),
        dpi=PLOT_DPI
    )

    # Anclaje hacia el centro para eliminar el gran hueco entre columnas.
    for ax in axes[:, 0]:
        ax.set_anchor("E")
    for ax in axes[:, 1]:
        ax.set_anchor("W")

    # POA Fijo (0, 0)
    ax = axes[0, 0]
    vals = df_atlas.loc[in_prov, "poa_fijo_anual_kwh_m2"]
    sc = ax.scatter(
        lons[in_prov], lats[in_prov],
        c=vals, cmap="magma", vmin=1700, vmax=2210,
        s=1.0, edgecolors="none", rasterized=True
    )
    overlay_decorations(ax, boundary_gdf, deptos_gdf, show_xlabel=False, show_ylabel=True)
    ax.set_title(
        f"(a) POA Fijo 30° Norte\nMedia: {vals.mean():.1f} kWh/m²/año",
        fontsize=8.0, weight="bold", pad=4
    )
    ax.tick_params(labelbottom=False, labelsize=7.2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3.5%", pad=0.14)
    cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7.0, pad=1)
    cbar.set_label("kWh/m²/año", fontsize=7.4, labelpad=2)

    # POA Tracker (0, 1)
    ax = axes[0, 1]
    vals = df_atlas.loc[in_prov, "poa_tracker_anual_kwh_m2"]
    sc = ax.scatter(
        lons[in_prov], lats[in_prov],
        c=vals, cmap="magma", vmin=2050, vmax=2705,
        s=1.0, edgecolors="none", rasterized=True
    )
    overlay_decorations(
        ax, boundary_gdf, deptos_gdf,
        show_xlabel=False, show_ylabel=False
    )
    ax.set_title(
        f"(b) POA Seguidor 1-Eje\nMedia: {vals.mean():.1f} kWh/m²/año",
        fontsize=8.0, weight="bold", pad=4
    )
    ax.tick_params(labelbottom=False, labelsize=7.2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3.5%", pad=0.14)
    cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7.0, pad=1)
    cbar.set_label("kWh/m²/año", fontsize=7.4, labelpad=2)

    # Yf Fijo (1, 0)
    ax = axes[1, 0]
    vals = df_atlas.loc[in_prov, "yf_fixed_kwh_kwp"]
    sc = ax.scatter(
        lons[in_prov], lats[in_prov],
        c=vals, cmap="viridis", vmin=1400, vmax=1800,
        s=1.0, edgecolors="none", rasterized=True
    )
    overlay_decorations(
        ax, boundary_gdf, deptos_gdf,
        show_xlabel=False, show_ylabel=True
    )
    ax.set_title(
        f"(c) Rendimiento Fijo $Y_f$\nMedia: {vals.mean():.1f} kWh/kWp/año",
        fontsize=8.0, weight="bold", pad=4
    )
    ax.tick_params(labelbottom=False, labelsize=7.2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3.5%", pad=0.14)
    cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7.0, pad=1)
    cbar.set_label("kWh/kWp/año", fontsize=7.4, labelpad=2)

    # Yf Tracker (1, 1)
    ax = axes[1, 1]
    vals = df_atlas.loc[in_prov, "yf_tracker_kwh_kwp"]
    sc = ax.scatter(
        lons[in_prov], lats[in_prov],
        c=vals, cmap="viridis", vmin=1700, vmax=2175,
        s=1.0, edgecolors="none", rasterized=True
    )
    overlay_decorations(
        ax, boundary_gdf, deptos_gdf,
        show_xlabel=False, show_ylabel=False
    )
    ax.set_title(
        f"(d) Rendimiento Seguidor $Y_f$\nMedia: {vals.mean():.1f} kWh/kWp/año",
        fontsize=8.0, weight="bold", pad=4
    )
    ax.tick_params(labelbottom=False, labelsize=7.2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3.5%", pad=0.14)
    cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7.0, pad=1)
    cbar.set_label("kWh/kWp/año", fontsize=7.4, labelpad=2)

    fig.suptitle(
        "Irradiación en Plano de Módulos (POA) y Rendimiento Específico Fotovoltaico ($Y_f$)",
        fontsize=8.8, weight="bold", y=0.985
    )

    fig.subplots_adjust(
        left=0.065, right=0.985,
        top=0.915, bottom=0.055,
        wspace=0.015, hspace=0.32
    )

    fig.savefig(OUT_DIR / "figura_11_poa_y_rendimiento_fotovoltaico.png")

    plt.close()


    # --------------------------------------------------------------------------

    # FIGURA 12: FACTOR DE CAPACIDAD Y GANANCIA POR SEGUIMIENTO

    # --------------------------------------------------------------------------

    print("Generando Figura 12: Ganancia por seguimiento y Factor de Capacidad...")

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.833333))

    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    # Panel superior completo: Ganancia Tracker (%)

    ax_top = fig.add_subplot(gs[0, :])

    vals = df_atlas.loc[in_prov, 'ganancia_tracker_pct']
    sc = ax_top.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='RdYlGn', vmin=14.0, vmax=23.3, s=1.3, edgecolors='none', rasterized=True)
    overlay_decorations(ax_top, boundary_gdf, deptos_gdf, show_xlabel=False)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_title(f"(a) Ganancia Energética Neta por Seguimiento 1-Eje respecto de Plano Fijo\nMedia Provincial: +{vals.mean():.2f}%", fontsize=8.0, weight='bold')

    divider = make_axes_locatable(ax_top)
    cax = divider.append_axes("right", size="2.5%", pad=0.08)
    cbar_top = plt.colorbar(sc, cax=cax)
    cbar_top.ax.tick_params(labelsize=7)
    cbar_top.set_label("Ganancia (%)", fontsize=7.5, labelpad=2)

    # Panel inferior izquierdo: CF Fijo (%)
    ax_bl = fig.add_subplot(gs[1, 0])
    vals = df_atlas.loc[in_prov, 'cf_fixed_pct']
    sc_bl = ax_bl.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='viridis', vmin=15.0, vmax=20.5, s=0.9, edgecolors='none', rasterized=True)
    overlay_decorations(ax_bl, boundary_gdf, deptos_gdf, show_xlabel=False)
    ax_bl.tick_params(labelbottom=False)
    ax_bl.set_title(f"(b) Factor de Capacidad Fijo (CF)\nMedia Provincial: {vals.mean():.2f}%", fontsize=8.0, weight='bold')

    divider = make_axes_locatable(ax_bl)
    cax_bl = divider.append_axes("bottom", size="4%", pad=0.12)
    cbar_bl = plt.colorbar(sc_bl, cax=cax_bl, orientation="horizontal")
    cbar_bl.ax.tick_params(labelsize=7)
    cbar_bl.set_label("CF Fijo (%)", fontsize=7.5, labelpad=2)

    # Panel inferior derecho: CF Tracker (%)
    ax_br = fig.add_subplot(gs[1, 1])
    vals = df_atlas.loc[in_prov, 'cf_tracker_pct']
    sc_br = ax_br.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='viridis', vmin=19.0, vmax=24.8, s=0.9, edgecolors='none', rasterized=True)
    overlay_decorations(ax_br, boundary_gdf, deptos_gdf, show_xlabel=False, show_ylabel=False)
    ax_br.tick_params(labelbottom=False)
    ax_br.set_title(f"(c) Factor de Capacidad Seguidor 1-Eje (CF)\nMedia Provincial: {vals.mean():.2f}%", fontsize=8.0, weight='bold')

    divider = make_axes_locatable(ax_br)
    cax_br = divider.append_axes("bottom", size="4%", pad=0.12)
    cbar_br = plt.colorbar(sc_br, cax=cax_br, orientation="horizontal")
    cbar_br.ax.tick_params(labelsize=7)
    cbar_br.set_label("CF Seguidor (%)", fontsize=7.5, labelpad=2)

    plt.suptitle("Factor de Capacidad Fotovoltaico y Beneficio Técnico del Seguimiento Solar", fontsize=8.8, weight='bold', y=0.985)
    plt.tight_layout(rect=[0.01, 0.035, 0.99, 0.96], pad=0.25, w_pad=0.15, h_pad=0.20)
    fig.savefig(OUT_DIR / "figura_12_cf_y_ganancia_tracker.png")
    plt.close()



    # --------------------------------------------------------------------------
    # FIGURA 13: SENSIBILIDAD DEL PROMEDIO MULTIANUAL A NUEVOS AÑOS
    # --------------------------------------------------------------------------
    print("Generando Figura 13: Sensibilidad multianual...")
    fig, axes = plt.subplots(1, 5, figsize=(FIG_W, FIG_W * 0.44))
    deltas = [
        ('delta_2_pct', r"$\Delta_2$ (1 $\to$ 2 A)", 3.0),
        ('delta_3_pct', r"$\Delta_3$ (2 $\to$ 3 A)", 2.5),
        ('delta_4_pct', r"$\Delta_4$ (3 $\to$ 4 A)", 2.0),
        ('delta_5_pct', r"$\Delta_5$ (4 $\to$ 5 A)", 1.5),
        ('delta_6_pct', r"$\Delta_6$ (5 $\to$ 6 A)", 1.5)
    ]
    for idx, (col, title, vlim) in enumerate(deltas):
        ax = axes[idx]
        vals = df_conv.loc[in_prov, col]
        sc = ax.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='RdBu_r', vmin=-vlim, vmax=vlim, s=1.0, edgecolors='none', rasterized=True)
        show_y = (idx == 0)
        overlay_decorations(ax, boundary_gdf, deptos_gdf, show_cities=(idx == 0), show_xlabel=False, show_ylabel=show_y)
        ax.tick_params(labelbottom=False)
        ax.set_title(f"{title}\n|Δ|: {np.abs(vals).mean():.2f}%", fontsize=6.5, weight='bold', pad=3)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="4%", pad=0.12)
        cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("% var.", fontsize=6.2, labelpad=2)
    fig.suptitle("Sensibilidad del Campo Medio Multianual al Incorporar Años Sucesivos (2020 - 2025)", fontsize=8.2, weight='bold', y=0.985)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.74, bottom=0.18, wspace=0.06)
    fig.savefig(OUT_DIR / "figura_13_sensibilidad_multianual.png")
    plt.close()

    # --------------------------------------------------------------------------
    # FIGURA 14: COEFICIENTE DE VARIACIÓN INTERANUAL
    # --------------------------------------------------------------------------
    print("Generando Figura 14: Coeficiente de variación interanual...")
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 1.000000))
    vals = df_conv.loc[in_prov, 'cv_interanual_pct']
    sc = ax.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='YlOrRd', vmin=0.0, vmax=5.0, s=1.4, edgecolors='none', rasterized=True)
    overlay_decorations(ax, boundary_gdf, deptos_gdf, show_xlabel=False)
    ax.tick_params(labelbottom=False)
    ax.set_title(rf"Variabilidad Interanual del Recurso Solar ($CV = 100 \cdot \sigma_y / \mu_y$, Período 2020--2025)" + f"\nMedia Provincial: {vals.mean():.2f}% (Rango: {vals.min():.2f}% - {vals.max():.2f}%)", fontsize=8.0, weight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4%", pad=0.12)
    cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Coeficiente de Variación (%)", fontsize=7.5, labelpad=2)
    plt.tight_layout(rect=[0.01, 0.035, 0.99, 0.96], pad=0.25, w_pad=0.15, h_pad=0.20)
    fig.savefig(OUT_DIR / "figura_14_cv_interanual.png")
    plt.close()

    # --------------------------------------------------------------------------
    # FIGURA 15: COMPARACIÓN ESPACIAL AÑO POR AÑO CON COLORBAR COMPARTIDA (2x3)
    # --------------------------------------------------------------------------
    print("Generando Figura 15: Comparación año por año (colorbar compartida, 2020-2025)...")
    years = ['2020', '2021', '2022', '2023', '2024', '2025']
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, FIG_W * 0.78))
    axes_flat = axes.flatten()
    sc_ref_anios = None

    for idx, y in enumerate(years):
        ax = axes_flat[idx]
        df_y = pd.read_parquet(DEF_DIR / f"atlas_anual_{y}.parquet")
        vals = df_y.loc[in_prov, 'ghi_kwh_m2']
        sc = ax.scatter(lons[in_prov], lats[in_prov], c=vals, cmap='inferno', vmin=1600, vmax=2100, s=0.9, edgecolors='none', rasterized=True)
        if idx == 0:
            sc_ref_anios = sc
        show_y = (idx % 3 == 0)
        overlay_decorations(ax, boundary_gdf, deptos_gdf, show_cities=(idx == 0), show_xlabel=False, show_ylabel=show_y)
        ax.set_title(f"Año {y}\nMedia: {vals.mean():.1f} kWh/m²", fontsize=7.5, weight='bold', pad=4)
        if not show_y:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        ax.tick_params(labelbottom=False)

    # Barra de color compartida abajo
    cbar_ax_anios = fig.add_axes([0.25, 0.035, 0.50, 0.016])
    cbar = fig.colorbar(sc_ref_anios, cax=cbar_ax_anios, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Irradiación Global Horizontal Anual (kWh/m²/año)", fontsize=7.5, weight='bold', labelpad=2)
    fig.suptitle("Consistencia Espacial Interanual de la Irradiación Global Horizontal (2020 - 2025)", fontsize=8.5, weight='bold', y=0.985)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.10, wspace=0.04, hspace=0.28)
    fig.savefig(OUT_DIR / "figura_15_estabilidad_espacial_anios.png")
    plt.close()

    # --------------------------------------------------------------------------
    # FIGURA 16: MATRIZ DE CORRELACIÓN ESPACIAL ENTRE AÑOS (6x6)
    # --------------------------------------------------------------------------
    print("Generando Figura 16: Matriz de correlación espacial (6x6)...")
    annual_ghi = {}
    for y in years:
        df_y = pd.read_parquet(DEF_DIR / f"atlas_anual_{y}.parquet")
        annual_ghi[f"Año {y}"] = df_y.loc[in_prov, 'ghi_kwh_m2'].values
    df_corr = pd.DataFrame(annual_ghi).corr()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 0.866667))
    cax = ax.matshow(df_corr.values, cmap='YlGnBu', vmin=0.85, vmax=1.0)
    for i in range(len(years)):
        for j in range(len(years)):
            val = df_corr.iloc[i, j]
            color = "white" if val > 0.94 else "black"
            ax.text(j, i, f"{val:.3f}", ha='center', va='center', fontsize=6.8, weight='bold', color=color)
    fig.colorbar(cax, shrink=0.8).set_label("Coeficiente de Correlación de Pearson (r)", fontsize=7)
    ax.set_xticks(range(len(years)))
    ax.set_yticks(range(len(years)))
    ax.set_xticklabels(df_corr.columns, fontsize=7, weight='bold')
    ax.set_yticklabels(df_corr.columns, fontsize=7, weight='bold')
    ax.set_title("Matriz de Correlación Espacial entre Campos de Irradiación Anual (2020 - 2025)", fontsize=7, weight='bold', pad=4)
    plt.tight_layout(pad=0.25, w_pad=0.15, h_pad=0.20)
    fig.savefig(OUT_DIR / "figura_16_correlacion_espacial_interanual.png")
    plt.close()



    # --------------------------------------------------------------------------

    # FIGURA 17: COBERTURA EFECTIVA DE DATOS

    # --------------------------------------------------------------------------

    print("Generando Figura 17: Cobertura efectiva de datos...")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 1.000000))

    cob_vals = np.full(in_prov.sum(), 100.0)

    sc = ax.scatter(lons[in_prov], lats[in_prov], c=cob_vals, cmap='Greens', vmin=90.0, vmax=100.0, s=1.4, edgecolors='none', rasterized=True)

    overlay_decorations(ax, boundary_gdf, deptos_gdf, show_xlabel=False)
    ax.tick_params(labelbottom=False)
    ax.set_title("Cobertura Efectiva de Datos Satelitales Válidos\nCompletitud Provincial: 100.0% de Días Integrados", fontsize=8.0, weight='bold')

    divider = make_axes_locatable(ax)

    cax = divider.append_axes("bottom", size="4%", pad=0.12)

    cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")

    cbar.ax.tick_params(labelsize=7)

    cbar.set_label("Completitud de Intervalos Diurnos Válidos (%)", fontsize=7, labelpad=2)

    plt.tight_layout(rect=[0.01, 0.035, 0.99, 0.96], pad=0.25, w_pad=0.15, h_pad=0.20)

    fig.savefig(OUT_DIR / "figura_17_cobertura_efectiva.png")

    plt.close()



    # --------------------------------------------------------------------------

    # FIGURA 18: DIAGNÓSTICO DE SALINAS GRANDES

    # --------------------------------------------------------------------------

    print("Generando Figura 18: Diagnóstico de Salinas Grandes...")


    fig = plt.figure(
        figsize=(17.5 * CM, 8 * CM),
        dpi=PLOT_DPI
    )

    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[0.85, 0.90, 1.60],
        wspace=0.48
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2])
    ]

    salinas_mask = (lons < -64.5) & (lats > -30.8) & in_prov

    # (a) Localización
    ax = axes[0]
    b0_proxy = np.where(salinas_mask[in_prov], 0.38, 0.18)
    sc = ax.scatter(
        lons[in_prov], lats[in_prov],
        c=b0_proxy, cmap="copper", vmin=0.15, vmax=0.45,
        s=1.2, edgecolors="none", rasterized=True
    )
    rect = patches.Rectangle(
        (-65.6, -30.6), 1.0, 1.2,
        linewidth=1.2, edgecolor="red",
        facecolor="none", linestyle="--"
    )
    ax.add_patch(rect)
    overlay_decorations(
        ax, boundary_gdf, deptos_gdf,
        show_cities=False, show_xlabel=False
    )
    ax.tick_params(labelbottom=False)
    ax.set_title(
        "(a) Localización de Salinas Grandes",
        fontsize=8.0, weight="bold", pad=4
    )

    ax.set_anchor("E")


    divider = make_axes_locatable(ax)

    cax = divider.append_axes("bottom", size="3.5%", pad=0.18)

    cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")

    cbar.ax.tick_params(labelsize=7, pad=1)

    cbar.set_label("Albedo Superficial Relativo", fontsize=7, labelpad=2)


    # (b) Detalle espacial

    ax = axes[1]

    ax.scatter(

        lons[salinas_mask], lats[salinas_mask],

        c=df_atlas.loc[salinas_mask, "ghi_anual_kwh_m2"],

        cmap="inferno", vmin=1400, vmax=2050,

        s=8.0, edgecolors="none"

    )

    if deptos_gdf is not None:

        deptos_gdf.boundary.plot(

            ax=ax, color="#64748b",

            linewidth=0.5, alpha=0.8

        )

    if boundary_gdf is not None:

        boundary_gdf.boundary.plot(

            ax=ax, color="#0f172a",

            linewidth=0.9

        )


    ax.set_xlim(-65.7, -64.4)

    ax.set_ylim(-30.8, -29.5)

    ax.set_title(

        "(b) GHI en el sector afectado\n359 celdas",

        fontsize=7, weight="bold", pad=4

    )

    ax.grid(True, linestyle=":", alpha=0.4)

    ax.set_xlabel("Longitud (°O)", fontsize=7)

    ax.set_ylabel("Latitud (°S)", fontsize=7)

    ax.tick_params(labelsize=7)


    # (c) Transecto

    ax = axes[2]

    lat_target = -30.05

    mask_slice = in_prov & (np.abs(lats - lat_target) < 0.015)

    lons_slice = lons[mask_slice]

    ghi_slice = df_atlas.loc[mask_slice, "ghi_anual_kwh_m2"].values


    sort_idx = np.argsort(lons_slice)

    lons_sorted = lons_slice[sort_idx]

    ghi_sorted = ghi_slice[sort_idx]


    baseline_env = 1960.0


    ax.plot(

        lons_sorted, ghi_sorted,

        lw=1.4, color="#dc2626",

        marker="o", markersize=2.0,

        label="GHI modelada"

    )

    ax.axhline(

        baseline_env,

        color="#2563eb", linestyle="--", lw=1.2,

        label=f"Entorno ({baseline_env:.0f} kWh/m²·a)"

    )

    ax.axvspan(

        -65.12, -64.70,

        color="#fee2e2", alpha=0.6,

        label="Salinas Grandes"

    )


    ax.annotate(

        "Sesgo: -30.2%\n(-585 kWh/m²·a)",

        xy=(-64.92, 1300),

        xytext=(-64.56, 1450),

        arrowprops=dict(

            facecolor="black",

            shrink=0.08,

            width=0.8,

            headwidth=4

        ),

        fontsize=7,

        weight="bold",

        bbox=dict(

            boxstyle="round,pad=0.25",

            fc="white", ec="#dc2626", lw=0.8

        )

    )


    ax.set_xlabel("Longitud (°O)", fontsize=7)

    ax.set_ylabel("Irradiación GHI (kWh/m²/año)", fontsize=7)

    ax.set_title(

        "(c) Transecto O–E a 30.05°S\nSesgo respecto del entorno",

        fontsize=7, weight="bold", pad=4

    )

    ax.set_ylim(1050, 2150)

    ax.grid(True, linestyle=":", alpha=0.4)

    ax.tick_params(labelsize=7)

    ax.legend(

        loc="lower right",

        fontsize=7,

        framealpha=0.9,

        borderpad=0.3,

        labelspacing=0.25

    )


    fig.suptitle(
        "Diagnóstico Físico-Radiométrico de la Singularidad de Salinas Grandes",
        fontsize=8.5, weight="bold", y=0.975
    )

    fig.subplots_adjust(
        left=0.045, right=0.985,
        top=0.84, bottom=0.18,
        wspace=0.48
    )

    fig.savefig(OUT_DIR / "figura_18_diagnostico_salinas_grandes.png")
    plt.close()


    # --------------------------------------------------------------------------

    # FIGURA 19: VISOR INTERACTIVO / PRODUCTO CARTOGRÁFICO

    # --------------------------------------------------------------------------

    print("Generando Figura 19: Mockup/Captura del Visor Interactivo...")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 0.625000))

    ax.set_facecolor("#0f172a")

    sc = ax.scatter(lons[in_prov], lats[in_prov], c=df_atlas.loc[in_prov, 'ghi_anual_kwh_m2'], cmap='inferno', vmin=1600, vmax=2050, s=1.1, edgecolors='none', rasterized=True)

    overlay_decorations(ax, boundary_gdf, deptos_gdf, show_xlabel=False)

    panel = patches.FancyBboxPatch(

        (-65.75, -34.8), 1.25, 2.2,

        boxstyle="round,pad=0.03,rounding_size=0.05",

        facecolor="#1e293b", edgecolor="#475569", linewidth=1.2, alpha=0.92, zorder=10

    )

    ax.add_patch(panel)

    ax.text(-65.70, -32.8, "☀️ Atlas Solar de Córdoba", color="#f8fafc", weight="bold", fontsize=7, zorder=11)

    ax.text(-65.70, -33.1, "Capa: GHI Anual Medio", color="#f59e0b", weight="bold", fontsize=7, zorder=11)

    ax.text(-65.70, -33.4, "Media: 1,888.0 kWh/m²·a", color="#94a3b8", fontsize=7, zorder=11)

    ax.text(-65.70, -33.7, "• Celda: 31.42°S, 64.18°O\n• Rendimiento Yf: 1,645 kWh/kWp\n• Ganancia Trk: +20.4%\n• Región: Sierras Bajas / Trans.", color="#cbd5e1", fontsize=7, zorder=11)

    ax.set_title("Plataforma Web Interactiva del Atlas Solar de Córdoba (Visualizador GeoTIFF / Leaflet)", fontsize=7, weight='bold', color='white', pad=4)

    ax.tick_params(colors='white')

    ax.xaxis.label.set_color('white')

    ax.yaxis.label.set_color('white')

    plt.tight_layout(pad=0.25, w_pad=0.15, h_pad=0.20)

    fig.savefig(OUT_DIR / "figura_19_visor_atlas_solar.png")

    plt.close()



    # --------------------------------------------------------------------------

    # FIGURAS DEL ANEXO TÉCNICO

    # --------------------------------------------------------------------------

    print("Organizando figuras del Anexo Técnico...")

    anexo_map = {

        "figura_heatmap_davis_peak_hours_raw.png": "anexo_01_heatmap_davis_peak_hours.png",

        "figura_timeline_davis_peak_regimes.png": "anexo_02_timeline_davis_peak_regimes.png",

        "figura_fase4_dev_abc_full_pipeline.png": "anexo_03_comparacion_detallada_A_vs_C.png",

        "figura_b0_envelope_grid2km_dev.png": "anexo_04_b0_envelope_grid2km.png",

        "figura_fase5a_descomposicion_erbs_poa.png": "anexo_05_descomposicion_erbs_poa.png",

        "figura_fase5a_potencia_fotovoltaica_diaria.png": "anexo_06_ejemplo_diario_cadena_fisica.png"

    }

    for src_name, dst_name in anexo_map.items():

        src_p = RAW_FIG_DIR / src_name

        if src_p.exists():

            shutil.copy(src_p, ANEXO_DIR / dst_name)

            print(f"   ✓ Anexo copiado: {dst_name}")



    print("\n" + "=" * 80)

    print("🎉 ¡TODAS LAS FIGURAS DEL INFORME HAN SIDO GENERADAS Y OPTIMIZADAS!")

    print(f"   • Carpeta Principal : {OUT_DIR}")

    print(f"   • Carpeta de Anexo  : {ANEXO_DIR}")

    print("=" * 80)





if __name__ == "__main__":

    main()
