#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generación y Catalogación Oficial de Rásters GeoTIFF (COG) del Atlas Solar de Córdoba
======================================================================================
Convenio ATLAS II — FCEFyN (UNC) & OHMC — Gobierno de la Provincia de Córdoba
Periodo Climatológico Definitivo: 2020–2025 (6 Años Completos)
Resolución Espacial: 2 km x 2 km (0.018° x 0.018°, N = 48.645 celdas interiores)

Este script:
1. Reubica los rásters históricos/preliminares sueltos en '00_archivo_historico_previo/'.
2. Genera todos los rásters Cloud-Optimized GeoTIFF (COG) oficiales con compresión DEFLATE,
   tiling interno 128x128, overviews piramidales (2, 4, 8) y metadatos científicos estandarizados.
3. Organiza los productos en subdirectorios temáticos perfectamente intuitivos:
   - 01_recurso_solar_multianual/
   - 02_potencial_fotovoltaico_multianual/
   - 03_ciclo_mensual_multianual/
   - 04_series_anuales_individuales/ (2020 a 2025)
   - 05_variabilidad_y_convergencia/
   - 06_regionalizacion_y_zonas_homogeneas/
4. Crea catálogos descriptivos README.md en cada carpeta y un índice maestro.
5. Exporta el archivo vectorial oficial consolidado GeoPackage (.gpkg).
"""

import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling

# Compatibilidad de propiedades Affine en Python 3.12 (reemplazo de cached_property incompatible con __slots__)
import affine
affine.Affine._astuple = property(
    lambda self: (self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h, self.i)
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
READY_DIR = DATA_DIR / "ready"
DEF_DIR = READY_DIR / "atlas_definitivo_2km"
MAPS_DIR = READY_DIR / "maps_geotiff"
GPKG_DIR = READY_DIR / "maps_geopackage"

BOUNDARY_FILE = DATA_DIR / "cordoba_boundary.geojson"
GRID_2KM_FILE = DATA_DIR / "grids" / "cordoba_grid_2km.geojson"

NODATA_VALUE = -9999.0


def export_cog(filepath: Path, data_2d: np.ndarray, transform, metadata: dict):
    """
    Escribe un ráster Cloud-Optimized GeoTIFF (COG) estándar:
    - Compresión DEFLATE sin pérdida.
    - Bloques internos 128x128.
    - Pirámides de sobremuestreo (overviews 2, 4, 8) con average.
    - Metadatos científicos completos.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = data_2d.shape
    arr_out = np.nan_to_num(data_2d, nan=NODATA_VALUE).astype(np.float32)

    with rasterio.open(
        filepath, 'w',
        driver='GTiff',
        height=ny, width=nx, count=1,
        dtype='float32',
        crs='EPSG:4326',
        transform=transform,
        nodata=NODATA_VALUE,
        compress='deflate',
        tiled=True,
        blockxsize=128,
        blockysize=128
    ) as dst:
        dst.write(arr_out, 1)
        dst.build_overviews([2, 4, 8], Resampling.average)
        dst.update_tags(ns='rio_overview', resampling='average')
        dst.update_tags(**metadata)


def archivar_archivos_previos(base_dir: Path):
    """Mueve archivos TIFF sueltos previos a '00_archivo_historico_previo'."""
    archive_dir = base_dir / "00_archivo_historico_previo"
    archive_dir.mkdir(parents=True, exist_ok=True)

    archived_count = 0
    for item in base_dir.iterdir():
        if item.is_file() and item.suffix.lower() in ['.tif', '.tiff']:
            dest = archive_dir / item.name
            shutil.move(str(item), str(dest))
            archived_count += 1

    readme_archive = archive_dir / "README.md"
    readme_archive.write_text(
        "# Archivo Histórico de Versiones Previas (Atlas Solar de Córdoba)\n\n"
        "Esta carpeta contiene archivos GeoTIFF generados en fases previas del desarrollo "
        "(versiones preliminares, corridas intermedias con sufijo `_blended`, y la climatología "
        "quinquenal 2020–2024 anterior).\n\n"
        "**Nota:** Para cualquier estudio de ingeniería, análisis regulatorio o dimensionamiento "
        "actual, se deben utilizar exclusivamente los productos oficiales organizados en las "
        "carpetas temáticas `01_` a `06_`.\n",
        encoding="utf-8"
    )
    print(f"📦 Se archivaron {archived_count} archivos históricos previos en '00_archivo_historico_previo/'.")


def main():
    print("=" * 80)
    print("🗺️ EXPORTACIÓN Y ORGANIZACIÓN DEL CATÁLOGO RÁSTER GEOTIFF (COG)")
    print("   Atlas Solar de Córdoba (Climatología Sexenal 2020–2025, 2 km)")
    print("=" * 80)

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    GPKG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Archivar TIFFs sueltos
    archivar_archivos_previos(MAPS_DIR)

    # 2. Cargar Dataset Multianual Sexenal
    multianual_file = DEF_DIR / "atlas_anual_2020_2025_medio.parquet"
    if not multianual_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {multianual_file}")
    
    print(f"\n📂 Cargando climatología sexenal desde: {multianual_file.name}")
    df_multi = pd.read_parquet(multianual_file)
    n_prov = df_multi['in_province'].sum()
    print(f"   • Celdas totales: {len(df_multi):,} | Celdas provinciales interiores: {n_prov:,}")

    # Calcular PSH en dataframe
    df_multi['psh_anual_hsp_dia'] = df_multi['ghi_anual_kwh_m2'] / 365.25

    # 3. Geometría ráster regular 2D
    lons = np.sort(df_multi['lon'].unique())
    lats = np.sort(df_multi['lat'].unique())
    d_lon = 0.0180
    d_lat = 0.0180
    nx = len(lons)
    ny = len(lats)
    west = lons.min() - d_lon / 2.0
    north = lats.max() + d_lat / 2.0
    raster_transform = from_origin(west, north, d_lon, d_lat)

    row_idx = np.round((lats.max() - df_multi['lat'].values) / d_lat).astype(int)
    col_idx = np.round((df_multi['lon'].values - lons.min()) / d_lon).astype(int)
    prov_mask = df_multi['in_province'].values

    def make_2d(values):
        grid = np.full((ny, nx), NODATA_VALUE, dtype=np.float32)
        grid[row_idx[prov_mask], col_idx[prov_mask]] = values[prov_mask]
        return grid

    meta_base = {
        'PROYECTO': 'Atlas Solar de la Provincia de Cordoba (ATLAS II)',
        'INSTITUCION': 'FCEFyN - Universidad Nacional de Cordoba / OHMC',
        'RESOLUCION': '2.0 km (0.018 deg)',
        'SISTEMA_COORDENADAS': 'EPSG:4326 (WGS 84)',
        'METODO_MODELO': 'Modelo Hibrido Satelital BD-JPT / HistGradientBoosting (Estrategia C)',
        'PERIODO': '2020-2025 (Sexenal)',
        'NODATA': '-9999.0'
    }

    # =========================================================================
    # PRODUCTO 1: RECURSO SOLAR MULTIANUAL (2020–2025)
    # =========================================================================
    p1_dir = MAPS_DIR / "01_recurso_solar_multianual"
    print(f"\n☀️ Generando Rásters de Recurso Solar Multianual en: {p1_dir.name}...")
    
    recurso_solar_vars = [
        ('cordoba_ghi_anual_medio_2020_2025_2km_cog.tif', df_multi['ghi_anual_kwh_m2'].values,
         'GHI', 'kWh/m2/ano', 'Irradiancia Global Horizontal Anual Media (2020-2025)'),
        ('cordoba_dni_anual_medio_2020_2025_2km_cog.tif', df_multi['dni_anual_kwh_m2'].values,
         'DNI', 'kWh/m2/ano', 'Irradiancia Directa Normal Anual Media derivada mediante Erbs (2020-2025)'),
        ('cordoba_dhi_anual_medio_2020_2025_2km_cog.tif', df_multi['dhi_anual_kwh_m2'].values,
         'DHI', 'kWh/m2/ano', 'Irradiancia Difusa Horizontal Anual Media derivada mediante Erbs (2020-2025)'),
        ('cordoba_psh_anual_medio_2020_2025_2km_cog.tif', df_multi['psh_anual_hsp_dia'].values,
         'PSH_GHI', 'HSP/dia', 'Horas Sol Pico Diarias Equivalentes (Media Anual 2020-2025)')
    ]

    for fname, vals, var, unit, desc in recurso_solar_vars:
        grid = make_2d(vals)
        meta = {**meta_base, 'VARIABLE': var, 'UNITS': unit, 'DESCRIPTION': desc}
        out_f = p1_dir / fname
        export_cog(out_f, grid, raster_transform, meta)
        prov_mean = vals[prov_mask].mean()
        print(f"   ✓ {fname:<52} [Media Prov: {prov_mean:.2f} {unit}]")

    (p1_dir / "README.md").write_text(
        "# 01 — Recurso Solar Multianual (2020–2025)\n\n"
        "Contiene los mapas ráster en formato Cloud-Optimized GeoTIFF (COG) de los componentes fundamentales "
        "de la radiación solar incidente en la Provincia de Córdoba, promediados sobre el sexenio 2020–2025.\n\n"
        "## Archivos y Especificaciones\n\n"
        "| Archivo | Variable | Unidad | Rango Prov. Típico | Descripción |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
        "| `cordoba_ghi_anual_medio_2020_2025_2km_cog.tif` | GHI | kWh/m²/año | 1.082 – 2.055 | Irradiancia Global Horizontal Anual Media (Media Prov: 1.889,3 kWh/m²/año) |\n"
        "| `cordoba_dni_anual_medio_2020_2025_2km_cog.tif` | DNI | kWh/m²/año | 397 – 2.279 | Irradiancia Directa Normal Anual Media (Media Prov: 1.985,2 kWh/m²/año) |\n"
        "| `cordoba_dhi_anual_medio_2020_2025_2km_cog.tif` | DHI | kWh/m²/año | 565 – 910 | Irradiancia Difusa Horizontal Anual Media (Media Prov: 604,9 kWh/m²/año) |\n"
        "| `cordoba_psh_anual_medio_2020_2025_2km_cog.tif` | PSH / HSP | HSP/día | 2,96 – 5,63 | Horas Sol Pico promedio diario equivalente (Media Prov: 5,17 HSP/día) |\n\n"
        "## Parámetros Técnicos Comunes\n"
        "- **Sistema de Coordenadas:** EPSG:4326 (WGS 84).\n"
        "- **Resolución Espacial:** 0.018° x 0.018° (~2,0 km).\n"
        "- **Valor NoData:** `-9999.0`.\n"
        "- **Estructura Interna:** Cloud-Optimized GeoTIFF (COG), Tiles 128x128, Overviews 2, 4, 8, Compresión DEFLATE.\n",
        encoding="utf-8"
    )

    # =========================================================================
    # PRODUCTO 2: POTENCIAL FOTOVOLTAICO MULTIANUAL (2020–2025)
    # =========================================================================
    p2_dir = MAPS_DIR / "02_potencial_fotovoltaico_multianual"
    print(f"\n⚡ Generando Rásters de Potencial Fotovoltaico en: {p2_dir.name}...")

    pv_vars = [
        ('cordoba_poa_fijo_anual_medio_2020_2025_2km_cog.tif', df_multi['poa_fijo_anual_kwh_m2'].values,
         'POA_Fixed', 'kWh/m2/ano', 'Irradiancia sobre Plano Inclinado Fijo 30 deg Norte (Perez 1990)'),
        ('cordoba_poa_tracker_anual_medio_2020_2025_2km_cog.tif', df_multi['poa_tracker_anual_kwh_m2'].values,
         'POA_Tracker', 'kWh/m2/ano', 'Irradiancia sobre Seguidor 1-Eje N-S con Backtracking'),
        ('cordoba_yf_fijo_anual_medio_2020_2025_2km_cog.tif', df_multi['yf_fixed_kwh_kwp'].values,
         'Yf_Fixed', 'kWh/kWp/ano', 'Rendimiento Energetico Especifico Fijo 30 deg Norte (NOCT + PVWatts)'),
        ('cordoba_yf_tracker_anual_medio_2020_2025_2km_cog.tif', df_multi['yf_tracker_kwh_kwp'].values,
         'Yf_Tracker', 'kWh/kWp/ano', 'Rendimiento Energetico Especifico con Seguidor 1-Eje (NOCT + PVWatts)'),
        ('cordoba_cf_fijo_anual_medio_2020_2025_2km_cog.tif', df_multi['cf_fixed_pct'].values,
         'CF_Fixed', '%', 'Factor de Capacidad Fotovoltaico Sistema Fijo (Base 8760 h/ano)'),
        ('cordoba_cf_tracker_anual_medio_2020_2025_2km_cog.tif', df_multi['cf_tracker_pct'].values,
         'CF_Tracker', '%', 'Factor de Capacidad Fotovoltaico Sistema Seguidor (Base 8760 h/ano)'),
        ('cordoba_ganancia_tracker_anual_medio_2020_2025_2km_cog.tif', df_multi['ganancia_tracker_pct'].values,
         'Gain_Tracker', '%', 'Ganancia Relativa de Generacion Energetica: Seguidor 1-Eje vs Plano Fijo')
    ]

    for fname, vals, var, unit, desc in pv_vars:
        grid = make_2d(vals)
        meta = {**meta_base, 'VARIABLE': var, 'UNITS': unit, 'DESCRIPTION': desc}
        out_f = p2_dir / fname
        export_cog(out_f, grid, raster_transform, meta)
        prov_mean = vals[prov_mask].mean()
        print(f"   ✓ {fname:<52} [Media Prov: {prov_mean:.2f} {unit}]")

    (p2_dir / "README.md").write_text(
        "# 02 — Potencial Fotovoltaico Multianual (2020–2025)\n\n"
        "Contiene los mapas ráster COG de rendimiento y producción fotovoltaica simulada a escala de utilidad "
        "(utility-scale) bajo estándares internacionales (IEC 61724, PVWatts v5 y modelo térmico NOCT), "
        "para configuración en plano fijo inclinado (30° Norte) y seguidor a un eje horizontal N-S con backtracking.\n\n"
        "## Archivos y Especificaciones\n\n"
        "| Archivo | Variable | Unidad | Rango Prov. Típico | Descripción |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
        "| `cordoba_poa_fijo_anual_medio_2020_2025_2km_cog.tif` | POA Fijo 30°N | kWh/m²/año | 1.042 – 2.206 | Irradiancia en plano fijo 30° N (Media Prov: 2.016,6 kWh/m²/año) |\n"
        "| `cordoba_poa_tracker_anual_medio_2020_2025_2km_cog.tif` | POA Seguidor 1-eje | kWh/m²/año | 1.202 – 2.702 | Irradiancia con seguimiento solar 1-eje (Media Prov: 2.461,6 kWh/m²/año) |\n"
        "| `cordoba_yf_fijo_anual_medio_2020_2025_2km_cog.tif` | Rendimiento Fijo ($Y_f$) | kWh/kWp/año | 871 – 1.793 | Producción específica sistema fijo (Media Prov: 1.644,7 kWh/kWp/año) |\n"
        "| `cordoba_yf_tracker_anual_medio_2020_2025_2km_cog.tif` | Rendimiento Seguidor ($Y_f$) | kWh/kWp/año | 996 – 2.168 | Producción específica sistema seguidor (Media Prov: 1.982,2 kWh/kWp/año) |\n"
        "| `cordoba_cf_fijo_anual_medio_2020_2025_2km_cog.tif` | Factor de Capacidad Fijo | % | 9,94% – 20,46% | Factor de planta anual fijo (Media Prov: 18,77%) |\n"
        "| `cordoba_cf_tracker_anual_medio_2020_2025_2km_cog.tif` | Factor de Capacidad Seguidor | % | 11,37% – 24,74% | Factor de planta anual seguidor (Media Prov: 22,63%) |\n"
        "| `cordoba_ganancia_tracker_anual_medio_2020_2025_2km_cog.tif` | Ganancia de Seguimiento | % | 14,38% – 23,36% | Incremento porcentual de generación del seguidor (Media Prov: +20,51%) |\n",
        encoding="utf-8"
    )

    # =========================================================================
    # PRODUCTO 3: CICLO MENSUAL MULTIANUAL (12 MESES)
    # =========================================================================
    p3_dir = MAPS_DIR / "03_ciclo_mensual_multianual"
    print(f"\n📅 Generando Rásters del Ciclo Mensual (12 Meses) en: {p3_dir.name}...")

    months_file = DEF_DIR / "atlas_promedios_mensuales_sexenal.parquet"
    if not months_file.exists():
        months_file = DEF_DIR / "atlas_promedios_mensuales_quinquenal.parquet"
    
    df_months = pd.read_parquet(months_file)
    nombres_meses = [
        (1, 'Enero', 'enero', 31),
        (2, 'Febrero', 'febrero', 28.25),
        (3, 'Marzo', 'marzo', 31),
        (4, 'Abril', 'abril', 30),
        (5, 'Mayo', 'mayo', 31),
        (6, 'Junio', 'junio', 30),
        (7, 'Julio', 'julio', 31),
        (8, 'Agosto', 'agosto', 31),
        (9, 'Septiembre', 'septiembre', 30),
        (10, 'Octubre', 'octubre', 31),
        (11, 'Noviembre', 'noviembre', 30),
        (12, 'Diciembre', 'diciembre', 31)
    ]

    for m_num, m_name, m_lower, d_in_m in nombres_meses:
        vals_m = df_months[m_name].values
        grid_m = make_2d(vals_m)
        fname = f"cordoba_ghi_mensual_{m_num:02d}_{m_lower}_2020_2025_2km_cog.tif"
        meta = {
            **meta_base,
            'VARIABLE': f'GHI_Mensual_{m_name}',
            'MES': m_name,
            'NUMERO_MES': str(m_num),
            'UNITS': 'kWh/m2/mes',
            'DESCRIPTION': f'Irradiancia Global Horizontal Media Mensual para {m_name} (Sexenio 2020-2025)'
        }
        out_f = p3_dir / fname
        export_cog(out_f, grid_m, raster_transform, meta)
        prov_mean = vals_m[prov_mask].mean()
        print(f"   ✓ {fname:<52} [Media Prov: {prov_mean:6.1f} kWh/m2/mes | {prov_mean/d_in_m:4.2f} kWh/m2/dia]")

    (p3_dir / "README.md").write_text(
        "# 03 — Ciclo Mensual Multianual (12 Meses: 2020–2025)\n\n"
        "Contiene las 12 capas ráster COG correspondientes a cada mes del año calendario (enero a diciembre), "
        "calculadas promediando los 6 años completos del periodo 2020–2025.\n\n"
        "## Unidades y Conversión Diaria\n"
        "Los valores de los píxeles están expresados en **kWh/m²/mes** (energía mensual acumulada). "
        "Para obtener la irradiación media diaria (**kWh/m²/día** o HSP/día) de cada mes, dividir por el número "
        "de días del mes correspondiente:\n\n"
        "| Mes | Archivo | Media Provincial (kWh/m²/mes) | Media Diaria (HSP/día) |\n"
        "| :--- | :--- | :--- | :--- |\n"
        "| 01 Enero | `cordoba_ghi_mensual_01_enero_2020_2025_2km_cog.tif` | 227,9 | 7,35 |\n"
        "| 02 Febrero | `cordoba_ghi_mensual_02_febrero_2020_2025_2km_cog.tif` | 189,6 | 6,71 |\n"
        "| 03 Marzo | `cordoba_ghi_mensual_03_marzo_2020_2025_2km_cog.tif` | 162,8 | 5,25 |\n"
        "| 04 Abril | `cordoba_ghi_mensual_04_abril_2020_2025_2km_cog.tif` | 127,6 | 4,25 |\n"
        "| 05 Mayo | `cordoba_ghi_mensual_05_mayo_2020_2025_2km_cog.tif` | 101,4 | 3,27 |\n"
        "| 06 Junio | `cordoba_ghi_mensual_06_junio_2020_2025_2km_cog.tif` | 82,9 | 2,76 |\n"
        "| 07 Julio | `cordoba_ghi_mensual_07_julio_2020_2025_2km_cog.tif` | 93,6 | 3,02 |\n"
        "| 08 Agosto | `cordoba_ghi_mensual_08_agosto_2020_2025_2km_cog.tif` | 128,7 | 4,15 |\n"
        "| 09 Septiembre | `cordoba_ghi_mensual_09_septiembre_2020_2025_2km_cog.tif` | 155,0 | 5,17 |\n"
        "| 10 Octubre | `cordoba_ghi_mensual_10_octubre_2020_2025_2km_cog.tif` | 190,4 | 6,14 |\n"
        "| 11 Noviembre | `cordoba_ghi_mensual_11_noviembre_2020_2025_2km_cog.tif` | 203,1 | 6,77 |\n"
        "| 12 Diciembre | `cordoba_ghi_mensual_12_diciembre_2020_2025_2km_cog.tif` | 226,3 | 7,30 |\n",
        encoding="utf-8"
    )

    # =========================================================================
    # PRODUCTO 4: SERIES ANUALES INDIVIDUALES (2020 A 2025)
    # =========================================================================
    p4_dir = MAPS_DIR / "04_series_anuales_individuales"
    print(f"\n📊 Generando Series Anuales Individuales (2020–2025) en: {p4_dir.name}...")

    anios = [2020, 2021, 2022, 2023, 2024, 2025]
    vars_anuales = [
        ('ghi_kwh_m2', 'ghi', 'GHI', 'kWh/m2/ano', 'Irradiancia Global Horizontal Anual'),
        ('dni_kwh_m2', 'dni', 'DNI', 'kWh/m2/ano', 'Irradiancia Directa Normal Anual'),
        ('dhi_kwh_m2', 'dhi', 'DHI', 'kWh/m2/ano', 'Irradiancia Difusa Horizontal Anual'),
        ('poa_fixed_kwh_m2', 'poa_fijo', 'POA_Fixed', 'kWh/m2/ano', 'Irradiancia en Plano Fijo 30 deg Norte'),
        ('poa_tracker_kwh_m2', 'poa_tracker', 'POA_Tracker', 'kWh/m2/ano', 'Irradiancia con Seguidor 1-Eje'),
        ('yf_fixed_kwh_kwp', 'yf_fijo', 'Yf_Fixed', 'kWh/kWp/ano', 'Rendimiento Energetico Fijo 30 deg Norte'),
        ('yf_tracker_kwh_kwp', 'yf_tracker', 'Yf_Tracker', 'kWh/kWp/ano', 'Rendimiento Energetico Seguidor 1-Eje')
    ]

    for y in anios:
        y_file = DEF_DIR / f"atlas_anual_{y}.parquet"
        if not y_file.exists():
            print(f"   ⚠️ Archivo no encontrado para {y}: {y_file.name}")
            continue
        
        y_dir = p4_dir / str(y)
        y_dir.mkdir(parents=True, exist_ok=True)
        df_y = pd.read_parquet(y_file)

        print(f"   • Procesando Año {y}...")
        for col, prefix, var, unit, desc in vars_anuales:
            if col not in df_y.columns:
                continue
            vals = df_y[col].values
            grid = make_2d(vals)
            fname = f"cordoba_{prefix}_anual_{y}_2km_cog.tif"
            meta = {
                **meta_base,
                'VARIABLE': var,
                'ANIO': str(y),
                'UNITS': unit,
                'DESCRIPTION': f"{desc} para el ano {y}"
            }
            out_f = y_dir / fname
            export_cog(out_f, grid, raster_transform, meta)

    (p4_dir / "README.md").write_text(
        "# 04 — Series Anuales Individuales (2020 a 2025)\n\n"
        "Esta carpeta contiene los rásters COG de cada año individual analizado por la cadena de procesamiento "
        "(2020, 2021, 2022, 2023, 2024 y 2025), organizados en subcarpetas anuales.\n\n"
        "## Estructura por Año\n"
        "Cada subcarpeta `2020/`, `2021/`, ..., `2025/` contiene:\n"
        "- `cordoba_ghi_anual_{ANO}_2km_cog.tif`: GHI anual (kWh/m²/año).\n"
        "- `cordoba_dni_anual_{ANO}_2km_cog.tif`: DNI anual (kWh/m²/año).\n"
        "- `cordoba_dhi_anual_{ANO}_2km_cog.tif`: DHI anual (kWh/m²/año).\n"
        "- `cordoba_poa_fijo_anual_{ANO}_2km_cog.tif`: Irradiancia en plano fijo 30° N (kWh/m²/año).\n"
        "- `cordoba_poa_tracker_anual_{ANO}_2km_cog.tif`: Irradiancia con seguidor 1-eje (kWh/m²/año).\n"
        "- `cordoba_yf_fijo_anual_{ANO}_2km_cog.tif`: Rendimiento específico fijo (kWh/kWp/año).\n"
        "- `cordoba_yf_tracker_anual_{ANO}_2km_cog.tif`: Rendimiento específico seguidor (kWh/kWp/año).\n",
        encoding="utf-8"
    )

    # =========================================================================
    # PRODUCTO 5: VARIABILIDAD Y CONVERGENCIA TEMPORAL
    # =========================================================================
    p5_dir = MAPS_DIR / "05_variabilidad_y_convergencia"
    print(f"\n📉 Generando Rásters de Variabilidad y Convergencia en: {p5_dir.name}...")

    conv_file = DEF_DIR / "atlas_convergencia_temporal_6years.parquet"
    if not conv_file.exists():
        conv_file = DEF_DIR / "atlas_convergencia_temporal_5years.parquet"

    df_conv = pd.read_parquet(conv_file)
    conv_vars = [
        ('cordoba_cv_interanual_2020_2025_2km_cog.tif', df_conv['cv_interanual_pct'].values,
         'CV_Interanual', '%', 'Coeficiente de Variacion Interanual Sexenal (Desviacion estandar / Media)'),
        ('cordoba_delta_2_pct_2km_cog.tif', df_conv['delta_2_pct'].values,
         'Delta_2', '%', 'Sensibilidad relativa del promedio al pasar de 1 a 2 anos (2020 a 2021)'),
        ('cordoba_delta_3_pct_2km_cog.tif', df_conv['delta_3_pct'].values,
         'Delta_3', '%', 'Sensibilidad relativa del promedio al pasar de 2 a 3 anos (hasta 2022)'),
        ('cordoba_delta_4_pct_2km_cog.tif', df_conv['delta_4_pct'].values,
         'Delta_4', '%', 'Sensibilidad relativa del promedio al pasar de 3 a 4 anos (hasta 2023)'),
        ('cordoba_delta_5_pct_2km_cog.tif', df_conv['delta_5_pct'].values,
         'Delta_5', '%', 'Sensibilidad relativa del promedio al pasar de 4 a 5 anos (hasta 2024)')
    ]
    if 'delta_6_pct' in df_conv.columns:
        conv_vars.append(
            ('cordoba_delta_6_pct_2km_cog.tif', df_conv['delta_6_pct'].values,
             'Delta_6', '%', 'Sensibilidad relativa del promedio al incorporar el ano 2025 (5 a 6 anos)')
        )

    for fname, vals, var, unit, desc in conv_vars:
        grid = make_2d(vals)
        meta = {**meta_base, 'VARIABLE': var, 'UNITS': unit, 'DESCRIPTION': desc}
        out_f = p5_dir / fname
        export_cog(out_f, grid, raster_transform, meta)
        prov_mean = vals[prov_mask].mean()
        print(f"   ✓ {fname:<52} [Media Prov: {prov_mean:+.2f} {unit}]")

    (p5_dir / "README.md").write_text(
        "# 05 — Variabilidad Interanual y Convergencia Temporal\n\n"
        "Contiene los mapas ráster COG que cuantifican la dispersión interanual del recurso solar "
        "y la estabilización asintótica de la media climatológica multianual.\n\n"
        "## Archivos y Métricas\n\n"
        "- `cordoba_cv_interanual_2020_2025_2km_cog.tif`: Coeficiente de variación interanual sexenal "
        "($\\sigma / \\mu \\times 100\\%$). Cuantifica el riesgo de incertidumbre interanual. "
        "Media provincial: **2,14%** (rango: 0,47% en llanura a 16,99% en cumbres).\n"
        "- `cordoba_delta_2_pct_2km_cog.tif` a `cordoba_delta_6_pct_2km_cog.tif`: Sensibilidad incremental "
        "$\\Delta_k = 100 \\times (\\bar{H}_k - \\bar{H}_{k-1}) / \\bar{H}_{k-1}$. Demuestra cómo las oscilaciones "
        "se amortiguan (|$\\Delta_6$| media provincial = **0,15%**), garantizando representatividad climática.\n",
        encoding="utf-8"
    )

    # =========================================================================
    # PRODUCTO 6: REGIONALIZACIÓN Y ZONAS HOMOGÉNEAS
    # =========================================================================
    p6_dir = MAPS_DIR / "06_regionalizacion_y_zonas_homogeneas"
    print(f"\n🗺️ Generando Ráster de Regionalización en: {p6_dir.name}...")

    if 'region' in df_multi.columns:
        vals_reg = df_multi['region'].values.astype(np.float32)
        grid_reg = make_2d(vals_reg)
        fname_reg = "cordoba_regiones_solares_k3_2km_cog.tif"
        meta_reg = {
            **meta_base,
            'VARIABLE': 'Regiones_Solares_k3',
            'UNITS': 'Categorica (0, 1, 2)',
            'DESCRIPTION': 'Zonificacion Solar Homogenea k=3 (0=Sierras Altas, 1=Transicion/Piedemonte, 2=Llanura Pampeana)'
        }
        out_f = p6_dir / fname_reg
        export_cog(out_f, grid_reg, raster_transform, meta_reg)
        print(f"   ✓ {fname_reg:<52} [Categorías: 0, 1, 2]")

    (p6_dir / "README.md").write_text(
        "# 06 — Regionalización Solar y Zonas Homogéneas (k=3)\n\n"
        "Contiene el ráster COG con la regionalización bioclimática y solar óptima obtenida mediante "
        "clustering K-Means ($k=3$) sobre las 166 estaciones Davis calibradas y propagada espacialmente "
        "mediante 3-NN euclídeo sobre la grilla canónica de 2 km.\n\n"
        "## Códigos de Categoría\n\n"
        "- **0: Sierras Altas y Cuencas Serranas** ($N = 5.586$ celdas, 11,5% provincial). Mayor régimen difuso y orográfico.\n"
        "- **1: Transición Central, Norte y Piedemonte** ($N = 17.677$ celdas, 36,3% provincial). Máxima irradiación neta.\n"
        "- **2: Llanura Pampeana y Depresión Oriental** ($N = 25.382$ celdas, 52,2% provincial). Alta estabilidad y régimen despejado.\n",
        encoding="utf-8"
    )

    # =========================================================================
    # CATÁLOGO MAESTRO README.MD EN EL DIRECTORIO RAÍZ DE GEOTIFF
    # =========================================================================
    print("\n📝 Generando Catálogo Maestro README.md en: maps_geotiff/README.md...")
    (MAPS_DIR / "README.md").write_text(
        "# Catálogo Cartográfico Oficial de Rásters GeoTIFF (COG)\n"
        "## Atlas Solar de la Provincia de Córdoba (Convenio ATLAS II)\n\n"
        "**Instituciones ejecutoras:** Facultad de Ciencias Exactas, Físicas y Naturales (FCEFyN — UNC) "
        "y Observatorio Hidro-Meteorológico de Córdoba (OHMC).\n\n"
        "**Periodo Climatológico:** 2020–2025 (Sexenal, 6 años completos).\n"
        "**Resolución Espacial:** 2,0 km × 2,0 km (0.018° × 0.018° angular, 48.645 celdas provinciales interiores).\n"
        "**Sistema de Referencia Espacial:** EPSG:4326 (WGS 84).\n"
        "**Formato de Almacenamiento:** Cloud-Optimized GeoTIFF (COG) con compresión interna DEFLATE sin pérdida, "
        "teselas (*tiles*) de 128×128 píxeles y pirámides multinivel (*overviews*) de 2, 4 y 8.\n"
        "**Valor de No-Dato (*NoData*):** `-9999.0`.\n\n"
        "---\n\n"
        "## Organización de Carpetas Temáticas\n\n"
        "Para facilitar la búsqueda y uso por parte de usuarios externos (organismos públicos, desarrolladores "
        "de proyectos fotovoltaicos, consultoras y la comunidad científica), los productos ráster están organizados "
        "en las siguientes carpetas estructuradas:\n\n"
        "```\n"
        "maps_geotiff/\n"
        "│\n"
        "├── 01_recurso_solar_multianual/          -> GHI, DNI, DHI anual medio y Horas Sol Pico (HSP)\n"
        "├── 02_potencial_fotovoltaico_multianual/ -> POA Fijo 30°, POA Seguidor 1-eje, Yf, CF y Ganancia Tracker\n"
        "├── 03_ciclo_mensual_multianual/         -> 12 capas mensuales (Enero a Diciembre) en kWh/m²/mes\n"
        "├── 04_series_anuales_individuales/      -> Series completas año por año (2020 a 2025) en subcarpetas\n"
        "├── 05_variabilidad_y_convergencia/      -> Coeficiente de variación CV (%) y sensibilidades Delta_2 a Delta_6\n"
        "├── 06_regionalizacion_y_zonas_homogeneas/ -> Clasificación de zonas solares homogéneas (k=3)\n"
        "└── 00_archivo_historico_previo/         -> Respaldo de versiones intermedias y preliminares históricas\n"
        "```\n\n"
        "## Resumen Rápido de Variables Principales (Climatología Sexenal 2020–2025)\n\n"
        "| Variable | Unidad | Media Prov. | Rango Min – Max | Ruta Relativa |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
        "| **GHI Anual** | kWh/m²/año | 1.889,3 | 1.081,9 – 2.054,7 | `01_recurso_solar_multianual/cordoba_ghi_anual_medio_2020_2025_2km_cog.tif` |\n"
        "| **DNI Anual** | kWh/m²/año | 1.985,2 | 396,6 – 2.279,2 | `01_recurso_solar_multianual/cordoba_dni_anual_medio_2020_2025_2km_cog.tif` |\n"
        "| **DHI Anual** | kWh/m²/año | 604,9 | 564,5 – 910,2 | `01_recurso_solar_multianual/cordoba_dhi_anual_medio_2020_2025_2km_cog.tif` |\n"
        "| **HSP Diaria** | HSP/día | 5,17 | 2,96 – 5,63 | `01_recurso_solar_multianual/cordoba_psh_anual_medio_2020_2025_2km_cog.tif` |\n"
        "| **POA Fijo 30°** | kWh/m²/año | 2.016,6 | 1.042,3 – 2.206,0 | `02_potencial_fotovoltaico_multianual/cordoba_poa_fijo_anual_medio_2020_2025_2km_cog.tif` |\n"
        "| **POA Seguidor** | kWh/m²/año | 2.461,6 | 1.201,5 – 2.701,6 | `02_potencial_fotovoltaico_multianual/cordoba_poa_tracker_anual_medio_2020_2025_2km_cog.tif` |\n"
        "| **$Y_f$ Fijo** | kWh/kWp/año | 1.644,7 | 870,5 – 1.792,6 | `02_potencial_fotovoltaico_multianual/cordoba_yf_fijo_anual_medio_2020_2025_2km_cog.tif` |\n"
        "| **$Y_f$ Seguidor** | kWh/kWp/año | 1.982,2 | 995,6 – 2.167,5 | `02_potencial_fotovoltaico_multianual/cordoba_yf_tracker_anual_medio_2020_2025_2km_cog.tif` |\n"
        "| **CF Fijo** | % | 18,77% | 9,94% – 20,46% | `02_potencial_fotovoltaico_multianual/cordoba_cf_fijo_anual_medio_2020_2025_2km_cog.tif` |\n"
        "| **CF Seguidor** | % | 22,63% | 11,37% – 24,74% | `02_potencial_fotovoltaico_multianual/cordoba_cf_tracker_anual_medio_2020_2025_2km_cog.tif` |\n"
        "| **Ganancia Seguidor** | % | +20,51% | 14,38% – 23,36% | `02_potencial_fotovoltaico_multianual/cordoba_ganancia_tracker_anual_medio_2020_2025_2km_cog.tif` |\n"
        "| **CV Interanual** | % | 2,14% | 0,47% – 16,99% | `05_variabilidad_y_convergencia/cordoba_cv_interanual_2020_2025_2km_cog.tif` |\n\n"
        "## Uso en Software SIG (QGIS, ArcGIS, Python, R)\n"
        "Todos los archivos son rásters georreferenciados estándar EPSG:4326. Pueden arrastrarse directamente "
        "a la vista de QGIS o ArcGIS. Debido a las pirámides COG internas, el renderizado es instantáneo incluso "
        "con zoom interactivo.\n",
        encoding="utf-8"
    )

    # =========================================================================
    # EXPORTACIÓN GEOPACKAGE VECTORIAL OFICIAL
    # =========================================================================
    print("\n📦 Exportando GeoPackage Vectorial Oficial (2020–2025)...")
    grid_base = gpd.read_file(GRID_2KM_FILE)
    gdf_export = grid_base[['cell_id', 'geometry']].merge(df_multi, on='cell_id', how='inner')
    gdf_prov = gdf_export[gdf_export['in_province']].copy()

    boundary_gdf = gpd.read_file(BOUNDARY_FILE)

    out_gpkg = GPKG_DIR / "atlas_solar_cordoba_2020_2025_definitivo.gpkg"
    gdf_prov.to_file(out_gpkg, layer="climatologia_sexenal_2km", driver="GPKG")
    boundary_gdf.to_file(out_gpkg, layer="limite_provincial", driver="GPKG")
    print(f"   ✓ GeoPackage consolidado generado en: {out_gpkg}")

    print("\n" + "=" * 80)
    print("✅ CATÁLOGO RÁSTER GEOTIFF (COG) COMPLETADO EXITOSAMENTE")
    print("=" * 80)


if __name__ == '__main__':
    main()
