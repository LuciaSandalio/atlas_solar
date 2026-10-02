#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 5B - Paso 05: Exportación Oficial de Rásters Cloud-Optimized GeoTIFF (COG)
y GeoPackage Vectorial para el Atlas Solar Preliminar de Córdoba (Años 2020, 2021 y 2023)
========================================================================================
Convenio ATLAS II — Provincia de Córdoba (Componente Solar)

1. Rásters Anuales Consolidados (Promedio Multianual 2020, 2021 y 2023):
   - GHI anual medio (kWh/m²/año)
   - DNI anual medio (kWh/m²/año)
   - DHI anual medio (kWh/m²/año)
   - PSH solar pico (kWh/m²/día)
   - POA Fijo 30° Norte (kWh/m²/año)
   - POA Seguidor 1-eje Backtracking (kWh/m²/año)
   - Rendimiento Fijo Yf (kWh/kWp/año)
   - Rendimiento Seguidor Yf (kWh/kWp/año)
   - Factor de Capacidad Fijo (%)
   - Factor de Capacidad Seguidor (%)
   - Ganancia de Seguimiento (%)
   - Regiones Solares Homogéneas (k=3)

2. Rásters Anuales Individuales (2020, 2021 y 2023):
   - GHI, DNI, Yf_fijo, Yf_tracker para cada año completo.

3. Rásters Mensuales (Enero a Julio):
   - GHI mensual medio en kWh/m²/mes para cada uno de los 7 meses procesados.

4. Rásters de Diagnóstico y Estabilidad:
   - Coeficiente de variación interanual CV (%)
   - Sensibilidad Delta 2 (%) y Delta 3 (%)

5. GeoPackage Oficial Vectorial:
   - Capa 'atlas_anual_puntos': Centroides con tabla de atributos completa (48,645 celdas).
   - Capa 'atlas_anual_poligonos': Celdas regulares de 2 km x 2 km.
   - Capa 'limite_provincial': Polígono oficial de la Provincia de Córdoba.
"""

import sys
import os
import json
from pathlib import Path

# Compatibilidad Affine en Python 3.12
import affine
affine.Affine._astuple = property(
    lambda self: (self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h, self.i)
)

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
READY_DIR = DATA_DIR / "ready"
PRELIM_DIR = READY_DIR / "atlas_preliminar_2km"
MAPS_GEOTIFF_DIR = READY_DIR / "maps_geotiff"
MAPS_GEOTIFF_DIR.mkdir(parents=True, exist_ok=True)
MAPS_GPKG_DIR = READY_DIR / "maps_geopackage"
MAPS_GPKG_DIR.mkdir(parents=True, exist_ok=True)

BOUNDARY_FILE = DATA_DIR / "cordoba_boundary.geojson"
GRID_2KM_FILE = DATA_DIR / "grids" / "cordoba_grid_2km.geojson"


def export_cog_raster(filename: Path, data_2d: np.ndarray, transform, var_metadata: dict):
    """
    Escribe un ráster Cloud-Optimized GeoTIFF (COG) profesional mediante rasterio:
    - Compresión DEFLATE sin pérdida.
    - Bloques internos tiled (128x128).
    - Pirámides de overviews multinivel (2, 4, 8).
    - Metadatos científicos completos.
    - Nodata = -9999.0
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


def main():
    print("=" * 85)
    print("🌍 FASE 5B-05: EXPORTACIÓN RÁSTER OFICIAL GEOTIFF/COG Y GEOPACKAGE 2 KM")
    print("   Convenio ATLAS II — Provincia de Córdoba (Años Completos 2020, 2021 y 2023)")
    print("=" * 85)

    # 1. Cargar datos del Atlas Preliminar Consolidado
    atlas_mean_file = PRELIM_DIR / "atlas_anual_2020_2021_2023_medio.parquet"
    if not atlas_mean_file.exists():
        raise FileNotFoundError(f"❌ No se encontró {atlas_mean_file}")

    print(f"\n📂 1. Cargando Atlas Preliminar desde {atlas_mean_file.name}...")
    df_mean = pd.read_parquet(atlas_mean_file)
    n_total = len(df_mean)
    df_prov = df_mean[df_mean['in_province']].copy()
    print(f"   • Celdas totales grilla: {n_total:,}")
    print(f"   • Celdas provinciales interiores: {len(df_prov):,}")

    # Calcular PSH en el dataframe
    df_mean['psh_ghi_kwh_m2_d'] = df_mean['ghi_anual_kwh_m2'] / 365.25

    # 2. Configurar la geometría ráster 2D regular a 2 km
    print("\n📐 2. Configurando geometría ráster regular 2D...")
    lons = np.sort(df_mean['lon'].unique())
    lats = np.sort(df_mean['lat'].unique())
    d_lon = 0.0180
    d_lat = 0.0180

    nx = len(lons)
    ny = len(lats)
    west = lons.min() - d_lon / 2.0
    north = lats.max() + d_lat / 2.0
    raster_transform = from_origin(west, north, d_lon, d_lat)

    row_idx = np.round((lats.max() - df_mean['lat'].values) / d_lat).astype(int)
    col_idx = np.round((df_mean['lon'].values - lons.min()) / d_lon).astype(int)
    prov_mask = df_mean['in_province'].values

    print(f"   • Dimensiones ráster: {ny} filas x {nx} columnas ({ny*nx:,} píxeles)")
    print(f"   • Resolución angular: {d_lon:.4f}° x {d_lat:.4f}° (~2.0 km)")
    print(f"   • Extensión geográfica: Oeste={west:.5f}°, Norte={north:.5f}°")
    print(f"   • Transform Affine: {raster_transform}")

    # 3. Exportar Rásters Multianuales Oficiales (10 mapas temáticos + PSH + Regiones)
    print("\n💾 3. Exportando Colección Cartográfica Oficial de Rásters COG (Promedio Multianual)...")

    multianual_vars = {
        'ghi_anual_kwh_m2': {
            'prefijo': 'cordoba_ghi_anual_2km',
            'meta': {'variable': 'GHI', 'units': 'kWh/m2/yr', 'description': 'Irradiancia Global Horizontal Anual Media (2020, 2021 y 2023)'}
        },
        'dni_anual_kwh_m2': {
            'prefijo': 'cordoba_dni_anual_2km',
            'meta': {'variable': 'DNI', 'units': 'kWh/m2/yr', 'description': 'Irradiancia Directa Normal Anual Media (2020, 2021 y 2023)'}
        },
        'dhi_anual_kwh_m2': {
            'prefijo': 'cordoba_dhi_anual_2km',
            'meta': {'variable': 'DHI', 'units': 'kWh/m2/yr', 'description': 'Irradiancia Difusa Horizontal Anual Media (2020, 2021 y 2023)'}
        },
        'psh_ghi_kwh_m2_d': {
            'prefijo': 'cordoba_psh_ghi_2km',
            'meta': {'variable': 'PSH_GHI', 'units': 'kWh/m2/day', 'description': 'Horas Solar Pico Global Horizontal Media Diaria'}
        },
        'poa_fixed_kwh_m2': {
            'prefijo': 'cordoba_poa_fijo_2km',
            'meta': {'variable': 'POA_Fixed', 'units': 'kWh/m2/yr', 'description': 'Irradiancia en Plano Inclinado Fijo 30 Norte Media (2020, 2021 y 2023)'}
        },
        'poa_tracker_kwh_m2': {
            'prefijo': 'cordoba_poa_tracker_2km',
            'meta': {'variable': 'POA_Tracker', 'units': 'kWh/m2/yr', 'description': 'Irradiancia con Seguidor 1-Eje Backtracking Media (2020, 2021 y 2023)'}
        },
        'yf_fixed_kwh_kwp': {
            'prefijo': 'cordoba_rendimiento_fv_fijo_yf_2km',
            'meta': {'variable': 'Yf_Fixed', 'units': 'kWh/kWp/yr', 'description': 'Rendimiento Especifico Fijo 30 Norte Medio (2020, 2021 y 2023)'}
        },
        'yf_tracker_kwh_kwp': {
            'prefijo': 'cordoba_rendimiento_fv_tracker_yf_2km',
            'meta': {'variable': 'Yf_Tracker', 'units': 'kWh/kWp/yr', 'description': 'Rendimiento Especifico Seguidor 1-Eje Medio (2020, 2021 y 2023)'}
        },
        'cf_fixed_pct': {
            'prefijo': 'cordoba_cf_fijo_pct_2km',
            'meta': {'variable': 'CF_Fixed', 'units': '%', 'description': 'Factor de Capacidad Fijo 30 Norte Medio (2020, 2021 y 2023)'}
        },
        'cf_tracker_pct': {
            'prefijo': 'cordoba_cf_tracker_pct_2km',
            'meta': {'variable': 'CF_Tracker', 'units': '%', 'description': 'Factor de Capacidad Seguidor 1-Eje Medio (2020, 2021 y 2023)'}
        },
        'ganancia_tracker_pct': {
            'prefijo': 'cordoba_ganancia_seguimiento_pct_2km',
            'meta': {'variable': 'Gain_Tracker', 'units': '%', 'description': 'Ganancia Relativa de Rendimiento por Seguimiento Media (2020, 2021 y 2023)'}
        },
        'region': {
            'prefijo': 'cordoba_regiones_solares_k3_2km',
            'meta': {'variable': 'Region_Solar_k3', 'units': 'cluster', 'description': 'Regiones Solares Homogeneas k=3 (0=Sierras Altas, 1=Sierras Bajas/Transicion, 2=Llanura Pampeana)'}
        }
    }

    base_meta = {
        'pipeline_version': '5B.3.0',
        'grid_resolution': '2 km (0.018 deg)',
        'reference_period': 'Años Completos 2020, 2021 y 2023',
        'crs': 'EPSG:4326',
        'source': 'Convenio ATLAS II - Provincia de Cordoba - FCEFyN/UNC & OHMC'
    }

    for var, cfg in multianual_vars.items():
        grid_2d = np.full((ny, nx), -9999.0, dtype=np.float32)
        vals_prov = df_mean.loc[prov_mask, var].values
        grid_2d[row_idx[prov_mask], col_idx[prov_mask]] = vals_prov

        meta_dict = {**base_meta, **cfg['meta']}
        
        # Exportar con sufijo _preliminar.tif
        out_tif_prelim = MAPS_GEOTIFF_DIR / f"{cfg['prefijo']}_preliminar.tif"
        export_cog_raster(out_tif_prelim, grid_2d, raster_transform, meta_dict)
        
        # También actualizar con sufijo _blended.tif para compatibilidad hacia atrás
        out_tif_blended = MAPS_GEOTIFF_DIR / f"{cfg['prefijo']}_blended.tif"
        export_cog_raster(out_tif_blended, grid_2d, raster_transform, meta_dict)

        print(f"   ✓ COG generado: {out_tif_prelim.name:<52} ({out_tif_prelim.stat().st_size / 1e3:6.1f} KB)")

    # 4. Exportar Rásters Anuales Individuales (2020, 2021 y 2023)
    print("\n📅 4. Exportando Rásters COG Anuales Individuales (2020, 2021 y 2023)...")
    for y in [2020, 2021, 2023]:
        file_y = PRELIM_DIR / f"atlas_anual_{y}.parquet"
        if not file_y.exists():
            continue
        df_y = pd.read_parquet(file_y)
        
        y_vars = {
            'ghi_kwh_m2': (f'cordoba_ghi_anual_{y}_2km.tif', 'GHI', 'kWh/m2/yr', f'GHI Anual {y}'),
            'dni_kwh_m2': (f'cordoba_dni_anual_{y}_2km.tif', 'DNI', 'kWh/m2/yr', f'DNI Anual {y}'),
            'yf_fixed_kwh_kwp': (f'cordoba_rendimiento_fijo_yf_{y}_2km.tif', 'Yf_Fixed', 'kWh/kWp/yr', f'Rendimiento Fijo Yf {y}'),
            'yf_tracker_kwh_kwp': (f'cordoba_rendimiento_tracker_yf_{y}_2km.tif', 'Yf_Tracker', 'kWh/kWp/yr', f'Rendimiento Seguidor Yf {y}')
        }
        for col_name, (fname, var_code, unit, desc) in y_vars.items():
            if col_name in df_y.columns:
                grid_2d = np.full((ny, nx), -9999.0, dtype=np.float32)
                vals_prov = df_y.loc[prov_mask, col_name].values
                grid_2d[row_idx[prov_mask], col_idx[prov_mask]] = vals_prov
                
                out_tif = MAPS_GEOTIFF_DIR / fname
                m_dict = {**base_meta, 'variable': var_code, 'units': unit, 'description': desc, 'year': str(y)}
                export_cog_raster(out_tif, grid_2d, raster_transform, m_dict)
                print(f"   ✓ COG {y}: {out_tif.name:<52} ({out_tif.stat().st_size / 1e3:6.1f} KB)")

    # 5. Exportar Rásters Mensuales (Enero -- Julio)
    print("\n🗓️ 5. Exportando Rásters COG Mensuales (Enero a Julio, Media Multianual)...")
    monthly_file = PRELIM_DIR / "atlas_promedios_mensuales_ene_jul.parquet"
    if monthly_file.exists():
        df_m = pd.read_parquet(monthly_file)
        month_names = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio']
        days_in_month = [31, 28.5, 31, 30, 31, 30, 31]
        
        for idx_m, m_name in enumerate(month_names, 1):
            if m_name in df_m.columns:
                grid_2d = np.full((ny, nx), -9999.0, dtype=np.float32)
                vals_prov = df_m.loc[prov_mask, m_name].values
                grid_2d[row_idx[prov_mask], col_idx[prov_mask]] = vals_prov
                
                # Ráster en kWh/m2/mes
                fname_mes = f"cordoba_ghi_mensual_{idx_m:02d}_{m_name.lower()}_2km.tif"
                out_tif_mes = MAPS_GEOTIFF_DIR / fname_mes
                m_dict = {**base_meta, 'variable': 'GHI_Monthly', 'units': 'kWh/m2/month', 'description': f'GHI Mensual Medio {m_name}', 'month': m_name}
                export_cog_raster(out_tif_mes, grid_2d, raster_transform, m_dict)
                
                # Ráster en kWh/m2/día
                grid_2d_dia = np.where(grid_2d > -9000, grid_2d / days_in_month[idx_m-1], -9999.0)
                fname_dia = f"cordoba_ghi_diario_{idx_m:02d}_{m_name.lower()}_2km.tif"
                out_tif_dia = MAPS_GEOTIFF_DIR / fname_dia
                m_dict_dia = {**base_meta, 'variable': 'GHI_Daily_Mean', 'units': 'kWh/m2/day', 'description': f'GHI Diario Medio {m_name}', 'month': m_name}
                export_cog_raster(out_tif_dia, grid_2d_dia, raster_transform, m_dict_dia)
                print(f"   ✓ COG Mes {idx_m:02d}: {out_tif_mes.name:<52} ({out_tif_mes.stat().st_size / 1e3:6.1f} KB)")

    # 6. Exportar Rásters de Diagnóstico: Variabilidad y Estabilidad
    print("\n📊 6. Exportando Rásters COG de Diagnóstico de Estabilidad...")
    conv_file = PRELIM_DIR / "atlas_convergencia_temporal.parquet"
    if conv_file.exists():
        df_c = pd.read_parquet(conv_file)
        diag_vars = {
            'cv_interanual_pct': ('cordoba_cv_interanual_pct_2km.tif', 'CV_Interanual', '%', 'Coeficiente de Variacion Interanual GHI (n=3)'),
            'delta_2_pct': ('cordoba_delta_2_pct_2km.tif', 'Delta_2', '%', 'Variacion Relativa al pasar de 1 a 2 Años (M20-21 vs M20)'),
            'delta_3_pct': ('cordoba_delta_3_pct_2km.tif', 'Delta_3', '%', 'Variacion Relativa al pasar de 2 a 3 Años (M20-21-23 vs M20-21)')
        }
        for col_name, (fname, var_code, unit, desc) in diag_vars.items():
            if col_name in df_c.columns:
                grid_2d = np.full((ny, nx), -9999.0, dtype=np.float32)
                vals_prov = df_c.loc[prov_mask, col_name].values
                grid_2d[row_idx[prov_mask], col_idx[prov_mask]] = vals_prov
                
                out_tif = MAPS_GEOTIFF_DIR / fname
                m_dict = {**base_meta, 'variable': var_code, 'units': unit, 'description': desc}
                export_cog_raster(out_tif, grid_2d, raster_transform, m_dict)
                print(f"   ✓ COG Diagnóstico: {out_tif.name:<52} ({out_tif.stat().st_size / 1e3:6.1f} KB)")

    # 7. Exportación a GeoPackage Vectorial Multipropósito
    print("\n📦 7. Exportando Paquete Vectorial GeoPackage Oficial...")
    if not BOUNDARY_FILE.exists():
        print(f"   ⚠️ Aviso: No se encontró {BOUNDARY_FILE}")
        boundary_gdf = None
    else:
        boundary_gdf = gpd.read_file(BOUNDARY_FILE)

    # Crear GeoDataFrame con puntos y polígonos
    df_export = df_mean[prov_mask].copy().reset_index(drop=True)
    
    # Capa 1: Centroides Puntos
    points_geom = gpd.points_from_xy(df_export['lon'], df_export['lat'], crs="EPSG:4326")
    gdf_points = gpd.GeoDataFrame(df_export, geometry=points_geom, crs="EPSG:4326")
    
    # Capa 2: Celdas Poligonales de 2 km x 2 km
    half_step = 0.009
    polys = [box(lon - half_step, lat - half_step, lon + half_step, lat + half_step) 
             for lon, lat in zip(df_export['lon'], df_export['lat'])]
    gdf_polys = gpd.GeoDataFrame(df_export, geometry=polys, crs="EPSG:4326")

    gpkg_paths = [
        MAPS_GPKG_DIR / "atlas_cordoba_2km_preliminar.gpkg",
        MAPS_GPKG_DIR / "atlas_cordoba_2km_blended.gpkg"
    ]

    for gpkg_path in gpkg_paths:
        print(f"   • Escribiendo en {gpkg_path.name}...")
        # Guardar capa de polígonos
        gdf_polys.to_file(gpkg_path, layer="atlas_anual_2km_poligonos", driver="GPKG")
        # Guardar capa de puntos centroides
        gdf_points.to_file(gpkg_path, layer="atlas_anual_2km_puntos", driver="GPKG")
        # Guardar capa de límite provincial
        if boundary_gdf is not None:
            boundary_gdf.to_file(gpkg_path, layer="limite_provincial", driver="GPKG")
        print(f"   ✓ GeoPackage completado: {gpkg_path.name} ({gpkg_path.stat().st_size / 1e6:5.2f} MB)")

    print("\n" + "=" * 85)
    print("🎉 ¡EXPORTACIÓN COMPLETA DE ENTREGABLES GEOTIFF/COG Y GEOPACKAGE EXITOSA!")
    print("=" * 85)


if __name__ == "__main__":
    main()
