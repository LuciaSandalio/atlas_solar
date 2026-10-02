#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 5B - Paso 04: Exportación Ráster Oficial (GeoTIFF/COG) y Cartografía Temática a 2 km
========================================================================================
1. Recibe directamente la climatología analítica nativa a 2 km generada por 5B-03
   (data/ready/atlas_climatology_annual_2km_{mode}.parquet y monthly).
   Sin interpolación intermedia ni suavizado espurio.
2. Enmascara y recorta estrictamente al límite territorial oficial de Córdoba
   (data/cordoba_boundary.geojson).
3. Construye rásters regulares 2D a partir del paso nativo de la grilla (0.018° ≈ 2 km)
   y exporta GeoTIFF estándar y Cloud-Optimized GeoTIFF (COG) con rasterio/GDAL:
   - Tiling por bloques (128x128).
   - Compresión DEFLATE sin pérdida.
   - Pirámides y overviews multinivel (2, 4, 8).
   - Metadatos científicos completos (unidades, método, nodata=-9999.0, CRS EPSG:4326).
4. Exporta GeoPackage vectorial multipropósito (data/ready/maps_geopackage/).
5. Genera cartografía temática de alta resolución (300 DPI) para el Informe 4 / Informe Final:
   - Mapa 1: Irradiación Global Horizontal Anual (GHI y PSH).
   - Mapa 2: Irradiación Directa Normal Anual (DNI).
   - Mapa 3: Rendimiento Específico en Plano Fijo 30° Norte (Yf).
   - Mapa 4: Rendimiento Específico con Seguidor 1-Eje Backtracking (Yf).
   - Mapa 5: Ganancia Relativa de Seguimiento Fotovoltaico (%).
   - Mapa 6: Diagnóstico de Estacionalidad Radiativa y Energética (Verano vs Invierno).
   - Mapa 7: Comparación Territorial Simultánea Fijo vs Seguidor 1-Eje.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Compatibilidad de Affine en Python 3.12
import affine
affine.Affine._astuple = property(
    lambda self: (self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h, self.i)
)

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Configuración visual para publicaciones científicas
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'grid.color': '#D0D0D0',
    'grid.linestyle': ':',
    'grid.linewidth': 0.6
})

DATA_DIR = Path("data")
READY_DIR = DATA_DIR / "ready"
MAPS_GEOTIFF_DIR = READY_DIR / "maps_geotiff"
MAPS_GEOTIFF_DIR.mkdir(parents=True, exist_ok=True)
MAPS_GPKG_DIR = READY_DIR / "maps_geopackage"
MAPS_GPKG_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("reportes/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY_FILE = DATA_DIR / "cordoba_boundary.geojson"
GRID_2KM_FILE = DATA_DIR / "grids" / "cordoba_grid_2km.geojson"

CIUDADES_CORDOBA = {
    "Córdoba Cap.": (-64.18, -31.42),
    "Río Cuarto": (-64.35, -33.12),
    "Villa María": (-63.24, -32.41),
    "San Francisco": (-62.08, -31.43),
    "Cruz del Eje": (-64.80, -30.72),
    "Villa Dolores": (-65.19, -31.94),
    "Mar Chiquita": (-62.68, -30.60)
}


def add_cities_and_boundary(ax, boundary_gdf):
    """Agrega el límite provincial oficial y las ciudades de referencia."""
    boundary_gdf.boundary.plot(ax=ax, color='#222222', linewidth=1.2, zorder=4)
    for name, (clon, clat) in CIUDADES_CORDOBA.items():
        ax.plot(clon, clat, 'ko', markersize=3.5, zorder=5)
        ax.text(clon + 0.07, clat - 0.05, name, fontsize=8, weight='semibold',
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8), zorder=6)


def export_cog_raster(filename: Path, data_2d: np.ndarray, transform, var_metadata: dict):
    """
    Escribe un ráster Cloud-Optimized GeoTIFF (COG) profesional mediante rasterio:
    - Compresión DEFLATE sin pérdida.
    - Bloques internos tiled (128x128).
    - Pirámides de overviews multinivel (2, 4, 8).
    - Metadatos científicos completos.
    """
    ny, nx = data_2d.shape
    arr_out = np.nan_to_num(data_2d, nan=-9999.0).astype(np.float32)

    with rasterio.open(
        filename, 'w',
        driver='GTiff',
        height=ny, width=nx, count=1,
        dtype='float32',
        crs='EPSG:4326',
        transform=transform,
        nodata=-9999.0,
        compress='deflate',
        tiled=True,
        blockxsize=128,
        blockysize=128
    ) as dst:
        dst.write(arr_out, 1)
        dst.build_overviews([2, 4, 8], Resampling.average)
        dst.update_tags(ns='rio_overview', resampling='average')
        dst.update_tags(**var_metadata)


def run_export_and_mapping(mode='blended'):
    print("=" * 80)
    print(f"🗺️ FASE 5B-04: EXPORTACIÓN RÁSTER OFICIAL Y CARTOGRAFÍA 2 KM (Modo: {mode.upper()})")
    print("=" * 80)

    # 1. Cargar climatologías anual y mensual oficiales a 2 km
    annual_file = READY_DIR / f"atlas_climatology_annual_2km_{mode}.parquet"
    monthly_file = READY_DIR / f"atlas_climatology_monthly_2km_{mode}.parquet"

    if not annual_file.exists():
        raise FileNotFoundError(f"❌ No se encontró climatología anual en {annual_file}. Ejecute 5B-03 primero.")

    print(f"📂 Cargando climatología anual oficial desde {annual_file}...")
    df_annual = pd.read_parquet(annual_file)
    n_total = len(df_annual)
    print(f"   • Celdas cargadas: {n_total:,}")

    # Cargar polígono provincial oficial
    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(f"❌ No se encontró límite oficial en {BOUNDARY_FILE}")
    boundary_gdf = gpd.read_file(BOUNDARY_FILE)
    print(f"   • Límite provincial oficial cargado: {BOUNDARY_FILE} (CRS: {boundary_gdf.crs})")

    # Filtrar celdas provinciales estrictas
    df_prov = df_annual[df_annual['in_province']].copy()
    print(f"   • Celdas dentro del territorio de Córdoba: {len(df_prov):,}")

    # 2. Reconstrucción de la matriz ráster 2D regular nativa
    print("\n📐 2. Configurando geometría ráster regular nativa a 2 km...")
    lons = np.sort(df_annual['lon'].unique())
    lats = np.sort(df_annual['lat'].unique())
    res_lon = np.median(np.diff(lons))
    res_lat = np.median(np.diff(lats))

    nx = len(lons)
    ny = len(lats)
    west = lons.min() - res_lon / 2.0
    north = lats.max() + res_lat / 2.0
    raster_transform = from_origin(west, north, res_lon, res_lat)

    row_idx = np.round((lats.max() - df_annual['lat'].values) / res_lat).astype(int)
    col_idx = np.round((df_annual['lon'].values - lons.min()) / res_lon).astype(int)

    prov_mask = df_annual['in_province'].values

    print(f"   • Dimensiones ráster: {ny} filas x {nx} columnas ({ny*nx:,} píxeles)")
    print(f"   • Resolución angular nativa: {res_lon:.4f}° x {res_lat:.4f}° (~2.0 km)")
    print(f"   • Extensión geográfica: Oeste={west:.4f}°, Norte={north:.4f}°")

    # 3. Exportación de GeoTIFF / COG Oficiales
    print("\n💾 3. Exportando rásters Cloud-Optimized GeoTIFF (COG) comprimidos...")

    raster_vars = {
        'ghi_kwh_m2_yr': {
            'filename': f'cordoba_ghi_anual_2km_{mode}.tif',
            'meta': {'variable': 'GHI', 'units': 'kWh/m2/yr', 'description': 'Irradiacion Global Horizontal Anual'}
        },
        'dni_kwh_m2_yr': {
            'filename': f'cordoba_dni_anual_2km_{mode}.tif',
            'meta': {'variable': 'DNI', 'units': 'kWh/m2/yr', 'description': 'Irradiacion Directa Normal Anual'}
        },
        'dhi_kwh_m2_yr': {
            'filename': f'cordoba_dhi_anual_2km_{mode}.tif',
            'meta': {'variable': 'DHI', 'units': 'kWh/m2/yr', 'description': 'Irradiacion Difusa Horizontal Anual'}
        },
        'psh_ghi_kwh_m2_d': {
            'filename': f'cordoba_psh_ghi_2km_{mode}.tif',
            'meta': {'variable': 'PSH_GHI', 'units': 'kWh/m2/day', 'description': 'Horas Solar Pico Global Horizontal'}
        },
        'poa_fixed_kwh_m2_yr': {
            'filename': f'cordoba_poa_fijo_2km_{mode}.tif',
            'meta': {'variable': 'POA_Fixed', 'units': 'kWh/m2/yr', 'description': 'Irradiacion en Plano Inclinado Fijo 30 Norte'}
        },
        'poa_tracker_kwh_m2_yr': {
            'filename': f'cordoba_poa_tracker_2km_{mode}.tif',
            'meta': {'variable': 'POA_Tracker', 'units': 'kWh/m2/yr', 'description': 'Irradiacion con Seguidor 1-Eje Backtracking'}
        },
        'yf_fixed_kwh_kwp_yr': {
            'filename': f'cordoba_rendimiento_fv_fijo_yf_2km_{mode}.tif',
            'meta': {'variable': 'Yf_Fixed', 'units': 'kWh/kWp/yr', 'description': 'Rendimiento Especifico Fijo 30 Norte'}
        },
        'yf_tracker_kwh_kwp_yr': {
            'filename': f'cordoba_rendimiento_fv_tracker_yf_2km_{mode}.tif',
            'meta': {'variable': 'Yf_Tracker', 'units': 'kWh/kWp/yr', 'description': 'Rendimiento Especifico Seguidor 1-Eje'}
        },
        'gain_energy_tracker_pct': {
            'filename': f'cordoba_ganancia_seguimiento_pct_2km_{mode}.tif',
            'meta': {'variable': 'Gain_Tracker', 'units': '%', 'description': 'Ganancia Relativa de Rendimiento por Seguimiento'}
        },
        'coverage_pct': {
            'filename': f'cordoba_cobertura_fisica_pct_2km_{mode}.tif',
            'meta': {'variable': 'Coverage', 'units': '%', 'description': 'Cobertura Temporal Fisica Efectiva'}
        }
    }

    for var, cfg in raster_vars.items():
        grid_2d = np.full((ny, nx), -9999.0, dtype=np.float32)
        # Asignar solo celdas provinciales (buffer recortado)
        vals_prov = df_annual.loc[prov_mask, var].values
        grid_2d[row_idx[prov_mask], col_idx[prov_mask]] = vals_prov

        out_tif = MAPS_GEOTIFF_DIR / cfg['filename']
        meta_dict = {
            **cfg['meta'],
            'pipeline_version': '5B.2.0',
            'grid_resolution': '2 km (0.018 deg)',
            'mode': mode,
            'source': 'Convenio ATLAS II - Provincia de Cordoba'
        }
        export_cog_raster(out_tif, grid_2d, raster_transform, meta_dict)
        print(f"   ✓ COG generado: {out_tif.name:<45} ({out_tif.stat().st_size / 1e6:4.2f} MB)")

    # 4. Exportación a GeoPackage Vectorial
    print("\n📦 4. Exportando capas vectoriales en GeoPackage...")
    grid_base = gpd.read_file(GRID_2KM_FILE)
    gdf_export = grid_base[['cell_id', 'geometry']].merge(df_annual, on='cell_id', how='inner')
    # Filtrar a provincia
    gdf_prov = gdf_export[gdf_export['in_province']].copy()

    out_gpkg = MAPS_GPKG_DIR / f"atlas_cordoba_2km_{mode}.gpkg"
    gdf_prov.to_file(out_gpkg, layer="climatologia_anual_2km", driver="GPKG")
    boundary_gdf.to_file(out_gpkg, layer="limite_provincial", driver="GPKG")
    print(f"   ✓ GeoPackage exportado con 2 capas en: {out_gpkg}")

    # 5. Generación de Cartografía Temática de Alta Resolución
    print("\n🎨 5. Generando mapas temáticos de alta resolución (300 DPI) para el informe...")

    def plot_thematic_map(var_name, title, cbar_label, cmap, filename, fmt="%.0f", vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(8, 9.5), constrained_layout=True)

        tri = ax.tricontourf(
            df_prov['lon'], df_prov['lat'], df_prov[var_name],
            levels=25, cmap=cmap, vmin=vmin, vmax=vmax, zorder=2
        )

        cs = ax.tricontour(
            df_prov['lon'], df_prov['lat'], df_prov[var_name],
            levels=8, colors='black', linewidths=0.4, alpha=0.5, vmin=vmin, vmax=vmax, zorder=3
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt=fmt)

        add_cities_and_boundary(ax, boundary_gdf)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.15)
        cbar = fig.colorbar(tri, cax=cax)
        cbar.set_label(cbar_label, weight='bold')

        ax.set_title(title, weight='bold', pad=12)
        ax.set_xlabel("Longitud (°O)", weight='bold')
        ax.set_ylabel("Latitud (°S)", weight='bold')
        ax.set_xlim(west, lons.max() + res_lon/2.0)
        ax.set_ylim(lats.min() - res_lat/2.0, north)
        ax.grid(True, zorder=1)

        out_img = FIGURES_DIR / filename
        fig.savefig(out_img)
        plt.close(fig)
        print(f"   ✓ Mapa generado: {out_img.name}")

    # Mapa 1: GHI Anual
    plot_thematic_map(
        'ghi_kwh_m2_yr',
        "Atlas Solar de Córdoba — Irradiación Global Horizontal (GHI)",
        "GHI Anual (kWh/m²·año)",
        'Spectral_r',
        'figura_fase5b_mapa1_ghi_anual.png',
        fmt="%.0f"
    )

    # Mapa 2: DNI Anual
    plot_thematic_map(
        'dni_kwh_m2_yr',
        "Atlas Solar de Córdoba — Irradiación Directa Normal (DNI)",
        "DNI Anual (kWh/m²·año)",
        'YlOrRd',
        'figura_fase5b_mapa2_dni_anual.png',
        fmt="%.0f"
    )

    # Mapa 3: Rendimiento FV Fijo 30° Norte
    plot_thematic_map(
        'yf_fixed_kwh_kwp_yr',
        "Atlas Solar de Córdoba — Rendimiento Específico Fijo ($Y_{f,\\mathrm{Fijo}}$)",
        "Rendimiento Específico (kWh/kWp·año)",
        'viridis',
        'figura_fase5b_mapa3_rendimiento_fijo.png',
        fmt="%.0f"
    )

    # Mapa 4: Rendimiento FV Seguidor 1-eje Backtracking
    plot_thematic_map(
        'yf_tracker_kwh_kwp_yr',
        "Atlas Solar de Córdoba — Rendimiento Específico con Seguidor 1-Eje ($Y_{f,\\mathrm{Trk}}$)",
        "Rendimiento Específico (kWh/kWp·año)",
        'viridis',
        'figura_fase5b_mapa4_rendimiento_tracker.png',
        fmt="%.0f"
    )

    # Mapa 5: Ganancia Relativa de Seguimiento (%)
    plot_thematic_map(
        'gain_energy_tracker_pct',
        "Atlas Solar de Córdoba — Ganancia Energética con Seguidor 1-Eje",
        "Ganancia de Rendimiento por Seguimiento (%)",
        'coolwarm',
        'figura_fase5b_mapa5_ganancia_tracker.png',
        fmt="%.1f%%"
    )

    # Mapa 6: Diagnóstico de Estacionalidad Radiativa Real (Verano vs Invierno)
    if monthly_file.exists():
        print("   • Generando Mapa 6: Diagnóstico de Estacionalidad Real (Verano vs Invierno)...")
        df_monthly = pd.read_parquet(monthly_file)
        # Meses de Verano: 12, 1, 2. Meses de Invierno: 6, 7, 8
        summer_months = [12, 1, 2]
        winter_months = [6, 7, 8]

        df_m_prov = df_monthly[df_monthly['in_province']].copy()
        summer_ghi = df_m_prov[df_m_prov['month'].isin(summer_months)].groupby('cell_id')['ghi_kwh_m2_mo'].sum().reset_index()
        winter_ghi = df_m_prov[df_m_prov['month'].isin(winter_months)].groupby('cell_id')['ghi_kwh_m2_mo'].sum().reset_index()

        df_seas = df_prov[['cell_id', 'lat', 'lon']].merge(summer_ghi.rename(columns={'ghi_kwh_m2_mo': 'ghi_verano'}), on='cell_id')
        df_seas = df_seas.merge(winter_ghi.rename(columns={'ghi_kwh_m2_mo': 'ghi_invierno'}), on='cell_id')
        df_seas['ratio_estacional'] = df_seas['ghi_verano'] / np.maximum(df_seas['ghi_invierno'], 1e-2)

        fig, axes = plt.subplots(1, 2, figsize=(15, 8.5), constrained_layout=True)

        for ax, col, title, cmap, label in zip(
            axes,
            ['ghi_verano', 'ghi_invierno'],
            ["(a) Verano Austral (Dic-Ene-Feb)", "(b) Invierno Austral (Jun-Jul-Ago)"],
            ['Spectral_r', 'Spectral_r'],
            ["GHI Verano (kWh/m²)", "GHI Invierno (kWh/m²)"]
        ):
            tri = ax.tricontourf(df_seas['lon'], df_seas['lat'], df_seas[col], levels=20, cmap=cmap, zorder=2)
            cs = ax.tricontour(df_seas['lon'], df_seas['lat'], df_seas[col], levels=6, colors='black', linewidths=0.4, alpha=0.5, zorder=3)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
            add_cities_and_boundary(ax, boundary_gdf)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.15)
            cbar = fig.colorbar(tri, cax=cax)
            cbar.set_label(label, weight='bold')
            ax.set_title(title, weight='bold', pad=10)
            ax.set_xlabel("Longitud (°O)", weight='bold')
            ax.set_ylabel("Latitud (°S)", weight='bold')
            ax.set_xlim(west, lons.max() + res_lon/2.0)
            ax.set_ylim(lats.min() - res_lat/2.0, north)
            ax.grid(True, zorder=1)

        fig.suptitle("Diagnóstico de Estacionalidad del Recurso Solar en Córdoba (Verano vs Invierno)", weight='bold', fontsize=15)
        out_img6 = FIGURES_DIR / "figura_fase5b_mapa6_estacionalidad_radiativa.png"
        fig.savefig(out_img6)
        plt.close(fig)
        print(f"   ✓ Mapa generado: {out_img6.name}")

    # Mapa 7: Comparativa Territorial Fijo vs Seguidor 1-Eje
    print("   • Generando Mapa 7: Comparativa Territorial Fijo vs Seguidor 1-Eje...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.5), constrained_layout=True)
    vmin_yf = min(df_prov['yf_fixed_kwh_kwp_yr'].min(), df_prov['yf_tracker_kwh_kwp_yr'].min())
    vmax_yf = max(df_prov['yf_fixed_kwh_kwp_yr'].max(), df_prov['yf_tracker_kwh_kwp_yr'].max())

    for ax, col, title in zip(
        axes,
        ['yf_fixed_kwh_kwp_yr', 'yf_tracker_kwh_kwp_yr'],
        ["(a) Estructura Fija (Tilt 30°, Azimut 0° Norte)", "(b) Seguidor 1-Eje Horizontal N-S (Backtracking)"]
    ):
        tri = ax.tricontourf(df_prov['lon'], df_prov['lat'], df_prov[col], levels=20, cmap='viridis', vmin=vmin_yf, vmax=vmax_yf, zorder=2)
        cs = ax.tricontour(df_prov['lon'], df_prov['lat'], df_prov[col], levels=8, colors='black', linewidths=0.4, alpha=0.5, vmin=vmin_yf, vmax=vmax_yf, zorder=3)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
        add_cities_and_boundary(ax, boundary_gdf)
        ax.set_title(title, weight='bold', pad=10)
        ax.set_xlabel("Longitud (°O)", weight='bold')
        ax.set_ylabel("Latitud (°S)", weight='bold')
        ax.set_xlim(west, lons.max() + res_lon/2.0)
        ax.set_ylim(lats.min() - res_lat/2.0, north)
        ax.grid(True, zorder=1)

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="4%", pad=0.15)
    cbar = fig.colorbar(tri, cax=cax)
    cbar.set_label("Rendimiento Específico $Y_f$ (kWh/kWp·año)", weight='bold')
    fig.suptitle("Comparación Territorial de Rendimiento Fotovoltaico: Fijo vs Seguidor 1-Eje", weight='bold', fontsize=15)
    out_img7 = FIGURES_DIR / "figura_fase5b_comparativa_fijo_tracker.png"
    fig.savefig(out_img7)
    plt.close(fig)
    print(f"   ✓ Mapa generado: {out_img7.name}")

    print("\n" + "=" * 80)
    print("✅ FASE 5B-04 COMPLETADA EXITOSAMENTE:")
    print(f"   • Rásters COG exportados en: {MAPS_GEOTIFF_DIR}")
    print(f"   • Capas vectoriales GPKG en:  {MAPS_GPKG_DIR}")
    print(f"   • Cartografía temática en:   {FIGURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exportación Ráster y Cartografía 2 km (Fase 5B-04).")
    parser.add_argument("--mode", choices=['blended', 'hard'], default='blended',
                        help="Modo de agregación insumo: 'blended' o 'hard'.")
    args = parser.parse_args()
    run_export_and_mapping(mode=args.mode)
