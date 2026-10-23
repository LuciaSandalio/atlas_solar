#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de Contenidos y Metadatos para el Visor Web del Atlas Solar (Fase 5B)
==============================================================================
Convenio ATLAS II — Provincia de Córdoba (FCEFyN/UNC & OHMC)

1. Procesa todos los GeoTIFFs en data/ready/maps_geotiff/
2. Genera overlays PNG RGBA transparentes para visualización rápida en Leaflet
3. Genera matrices de consulta de píxeles para interrogación interactiva (inspector)
4. Calcula histogramas y estadísticas completas (provinciales y por región k=3)
5. Exporta capas vectoriales departamentales y provinciales
6. Escribe viewer/data/manifest.json
"""

import sys
import os
import json
from pathlib import Path

# Compatibilidad Affine
import affine
affine.Affine._astuple = property(
    lambda self: (self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h, self.i)
)

import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
GEOTIFF_DIR = DATA_DIR / "ready" / "maps_geotiff"
VIEWER_DIR = ROOT_DIR / "viewer"
OVERLAYS_DIR = VIEWER_DIR / "overlays"
DATA_OUT_DIR = VIEWER_DIR / "data"
GRIDS_OUT_DIR = DATA_OUT_DIR / "grids"

OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
GRIDS_OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_hex_colors_from_cmap(cmap_name, n_stops=5):
    """Retorna una lista de colores hex para la leyenda."""
    cmap = plt.get_cmap(cmap_name)
    stops = []
    for i in range(n_stops):
        val = i / (n_stops - 1)
        rgba = cmap(val)
        hex_col = mcolors.to_hex(rgba)
        stops.append(hex_col)
    return stops


def main():
    # Do not silently certify the legacy scientific pipeline.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.atlas_contract import reject_legacy
    reject_legacy('Legacy viewer generation')

    print("=" * 80)
    print("🚀 PREPARANDO DATOS PARA EL VISOR WEB INTERACTIVO GEOTIFF")
    print("=" * 80)

    # 1. Cargar Regiones Solares k=3 para estadísticas zonales
    region_tif = GEOTIFF_DIR / "cordoba_regiones_solares_k3_2km_preliminar.tif"
    if not region_tif.exists():
        raise FileNotFoundError(f"❌ Falta {region_tif}")

    with rasterio.open(region_tif) as src:
        reg_data = src.read(1)
        ny, nx = reg_data.shape
        bounds = src.bounds
        nodata_val = src.nodata

    # Bounds Leaflet: [[south, west], [north, east]]
    leaflet_bounds = [
        [float(bounds.bottom), float(bounds.left)],
        [float(bounds.top), float(bounds.right)]
    ]
    print(f"📐 Extensión geográfica: {leaflet_bounds}")
    print(f"📐 Dimensiones matriz: {ny} filas x {nx} columnas")

    # 2. Exportar límites vectoriales
    print("\n📦 Exportando límites departamentales y provincial...")
    cba_deptos_shp = ROOT_DIR / "shapes_cba" / "cba_deptos.shp"
    if cba_deptos_shp.exists():
        gdf_deptos = gpd.read_file(cba_deptos_shp)
        if gdf_deptos.crs != "EPSG:4326":
            gdf_deptos = gdf_deptos.to_crs("EPSG:4326")
        gdf_deptos = gdf_deptos[['DEPARTAMTO', 'CABECERA', 'geometry']].copy()
        gdf_deptos['geometry'] = gdf_deptos['geometry'].simplify(0.002)
        deptos_json_path = DATA_OUT_DIR / "cordoba_deptos.geojson"
        gdf_deptos.to_file(deptos_json_path, driver="GeoJSON")
        print(f"   ✓ Departamentos exportados a {deptos_json_path.name} ({len(gdf_deptos)} deptos)")

    boundary_geojson = DATA_DIR / "cordoba_boundary.geojson"
    if boundary_geojson.exists():
        gdf_bound = gpd.read_file(boundary_geojson)
        if gdf_bound.crs != "EPSG:4326":
            gdf_bound = gdf_bound.to_crs("EPSG:4326")
        bound_json_path = DATA_OUT_DIR / "cordoba_boundary.geojson"
        gdf_bound.to_file(bound_json_path, driver="GeoJSON")
        print(f"   ✓ Límite provincial exportado a {bound_json_path.name}")

    # 3. Definición de Capas del Catálogo
    layers_config = [
        # --- MULTIANUAL CONSOLIDADO (10 MAPAS + PSH + REGIONES) ---
        {
            "id": "ghi_anual_preliminar",
            "tif": "cordoba_ghi_anual_2km_preliminar.tif",
            "name": "Irradiancia Global Horizontal (GHI)",
            "category": "Multianual Consolidado (2020, 2021, 2023)",
            "unit": "kWh/m²·año",
            "cmap": "inferno",
            "desc": "Energía solar total incidente sobre una superficie horizontal en el año promedio.",
            "format": ".1f"
        },
        {
            "id": "dni_anual_preliminar",
            "tif": "cordoba_dni_anual_2km_preliminar.tif",
            "name": "Irradiancia Directa Normal (DNI)",
            "category": "Multianual Consolidado (2020, 2021, 2023)",
            "unit": "kWh/m²·año",
            "cmap": "plasma",
            "desc": "Energía directa perpendicular al haz solar; esencial para seguidores solares y concentración.",
            "format": ".1f"
        },
        {
            "id": "dhi_anual_preliminar",
            "tif": "cordoba_dhi_anual_2km_preliminar.tif",
            "name": "Irradiancia Difusa Horizontal (DHI)",
            "category": "Multianual Consolidado (2020, 2021, 2023)",
            "unit": "kWh/m²·año",
            "cmap": "cividis",
            "desc": "Fracción difusa de la radiación dispersada por la atmósfera y las nubes.",
            "format": ".1f"
        },
        {
            "id": "psh_ghi_preliminar",
            "tif": "cordoba_psh_ghi_2km_preliminar.tif",
            "name": "Horas Solar Pico Diarias (PSH)",
            "category": "Multianual Consolidado (2020, 2021, 2023)",
            "unit": "kWh/m²/día",
            "cmap": "inferno",
            "desc": "Equivalente diario en horas a 1000 W/m² (HSP = GHI / 365.25).",
            "format": ".2f"
        },
        {
            "id": "poa_fijo_preliminar",
            "tif": "cordoba_poa_fijo_2km_preliminar.tif",
            "name": "Irradiancia en Plano Fijo 30° Norte (POA)",
            "category": "Multianual Consolidado (2020, 2021, 2023)",
            "unit": "kWh/m²·año",
            "cmap": "magma",
            "desc": "Radiación total incidente sobre módulos fotovoltaicos fijos orientados al Norte ecuatorial.",
            "format": ".1f"
        },
        {
            "id": "poa_tracker_preliminar",
            "tif": "cordoba_poa_tracker_2km_preliminar.tif",
            "name": "Irradiancia con Seguidor 1-Eje (POA)",
            "category": "Multianual Consolidado (2020, 2021, 2023)",
            "unit": "kWh/m²·año",
            "cmap": "magma",
            "desc": "Radiación total incidente sobre paneles con seguimiento solar 1-eje N-S y backtracking.",
            "format": ".1f"
        },
        {
            "id": "yf_fijo_preliminar",
            "tif": "cordoba_rendimiento_fv_fijo_yf_2km_preliminar.tif",
            "name": "Rendimiento Específico Fijo (Yf)",
            "category": "Multianual Consolidado (2020, 2021, 2023)",
            "unit": "kWh/kWp·año",
            "cmap": "viridis",
            "desc": "Generación eléctrica anual neta en corriente alterna inyectada a la red por cada kWp instalado.",
            "format": ".1f"
        },
        {
            "id": "yf_tracker_preliminar",
            "tif": "cordoba_rendimiento_fv_tracker_yf_2km_preliminar.tif",
            "name": "Rendimiento Específico Seguidor (Yf)",
            "category": "Multianual Consolidado (2020, 2021, 2023)",
            "unit": "kWh/kWp·año",
            "cmap": "viridis",
            "desc": "Generación neta AC en plantas equipadas con seguidores horizontales en 1 eje.",
            "format": ".1f"
        },
        {
            "id": "cf_fijo_preliminar",
            "tif": "cordoba_cf_fijo_pct_2km_preliminar.tif",
            "name": "Factor de Capacidad Fijo (CF)",
            "category": "Multianual Consolidado (2020, 2021, 2023)",
            "unit": "%",
            "cmap": "viridis",
            "desc": "Factor de planta porcentual anual para tecnología fotovoltaica fija (CF = Yf / 8760).",
            "format": ".1f"
        },
        {
            "id": "cf_tracker_preliminar",
            "tif": "cordoba_cf_tracker_pct_2km_preliminar.tif",
            "name": "Factor de Capacidad Seguidor (CF)",
            "category": "Multianual Consolidado (2020, 2021, 2023)",
            "unit": "%",
            "cmap": "viridis",
            "desc": "Factor de planta anual para parques fotovoltaicos con seguimiento en 1 eje.",
            "format": ".1f"
        },
        {
            "id": "ganancia_tracker_preliminar",
            "tif": "cordoba_ganancia_seguimiento_pct_2km_preliminar.tif",
            "name": "Ganancia por Seguimiento Solar",
            "category": "Multianual Consolidado (2020, 2021, 2023)",
            "unit": "%",
            "cmap": "RdYlGn",
            "desc": "Incremento relativo porcentual en generación eléctrica al incorporar seguimiento 1-eje.",
            "format": ".2f"
        },
        {
            "id": "regiones_k3",
            "tif": "cordoba_regiones_solares_k3_2km_preliminar.tif",
            "name": "Regiones Solares Homogéneas (k=3)",
            "category": "Multianual Consolidado (2020, 2021, 2023)",
            "unit": "cluster",
            "cmap": "tab10",
            "desc": "Provincias radiativas: 0=Sierras Altas (11.5%), 1=Sierras Bajas/Transición (36.3%), 2=Llanura Pampeana (52.2%).",
            "format": ".0f",
            "is_categorical": True
        },
        # --- AÑOS INDIVIDUALES ---
        {
            "id": "ghi_2020",
            "tif": "cordoba_ghi_anual_2020_2km.tif",
            "name": "GHI Anual 2020",
            "category": "Años Individuales",
            "unit": "kWh/m²·año",
            "cmap": "inferno",
            "desc": "Irradiancia global horizontal anual en 2020 (año bisiesto La Niña débil).",
            "format": ".1f"
        },
        {
            "id": "ghi_2021",
            "tif": "cordoba_ghi_anual_2021_2km.tif",
            "name": "GHI Anual 2021",
            "category": "Años Individuales",
            "unit": "kWh/m²·año",
            "cmap": "inferno",
            "desc": "Irradiancia global horizontal anual en 2021 (La Niña moderada).",
            "format": ".1f"
        },
        {
            "id": "ghi_2023",
            "tif": "cordoba_ghi_anual_2023_2km.tif",
            "name": "GHI Anual 2023",
            "category": "Años Individuales",
            "unit": "kWh/m²·año",
            "cmap": "inferno",
            "desc": "Irradiancia global horizontal anual en 2023 (transición a El Niño).",
            "format": ".1f"
        },
        {
            "id": "yf_tracker_2020",
            "tif": "cordoba_rendimiento_tracker_yf_2020_2km.tif",
            "name": "Rendimiento Seguidor Yf 2020",
            "category": "Años Individuales",
            "unit": "kWh/kWp·año",
            "cmap": "viridis",
            "desc": "Rendimiento específico anual con seguidor solar en 2020.",
            "format": ".1f"
        },
        {
            "id": "yf_tracker_2021",
            "tif": "cordoba_rendimiento_tracker_yf_2021_2km.tif",
            "name": "Rendimiento Seguidor Yf 2021",
            "category": "Años Individuales",
            "unit": "kWh/kWp·año",
            "cmap": "viridis",
            "desc": "Rendimiento específico anual con seguidor solar en 2021.",
            "format": ".1f"
        },
        {
            "id": "yf_tracker_2023",
            "tif": "cordoba_rendimiento_tracker_yf_2023_2km.tif",
            "name": "Rendimiento Seguidor Yf 2023",
            "category": "Años Individuales",
            "unit": "kWh/kWp·año",
            "cmap": "viridis",
            "desc": "Rendimiento específico anual con seguidor solar en 2023.",
            "format": ".1f"
        },
        # --- CICLO MENSUAL (MEDIAS DIARIAS ENE - JUL) ---
        {
            "id": "ghi_dia_01_ene",
            "tif": "cordoba_ghi_diario_01_enero_2km.tif",
            "name": "Enero — GHI Medio Diario",
            "category": "Ciclo Estacional Mensual (kWh/m²/día)",
            "unit": "kWh/m²/día",
            "cmap": "inferno",
            "desc": "Media diaria mensual de irradiancia global horizontal en Enero (Verano pleno).",
            "format": ".2f",
            "fixed_range": [2.0, 8.0]
        },
        {
            "id": "ghi_dia_02_feb",
            "tif": "cordoba_ghi_diario_02_febrero_2km.tif",
            "name": "Febrero — GHI Medio Diario",
            "category": "Ciclo Estacional Mensual (kWh/m²/día)",
            "unit": "kWh/m²/día",
            "cmap": "inferno",
            "desc": "Media diaria mensual de irradiancia global horizontal en Febrero.",
            "format": ".2f",
            "fixed_range": [2.0, 8.0]
        },
        {
            "id": "ghi_dia_03_mar",
            "tif": "cordoba_ghi_diario_03_marzo_2km.tif",
            "name": "Marzo — GHI Medio Diario",
            "category": "Ciclo Estacional Mensual (kWh/m²/día)",
            "unit": "kWh/m²/día",
            "cmap": "inferno",
            "desc": "Media diaria mensual de irradiancia global horizontal en Marzo (Equinoccio).",
            "format": ".2f",
            "fixed_range": [2.0, 8.0]
        },
        {
            "id": "ghi_dia_04_abr",
            "tif": "cordoba_ghi_diario_04_abril_2km.tif",
            "name": "Abril — GHI Medio Diario",
            "category": "Ciclo Estacional Mensual (kWh/m²/día)",
            "unit": "kWh/m²/día",
            "cmap": "inferno",
            "desc": "Media diaria mensual de irradiancia global horizontal en Abril.",
            "format": ".2f",
            "fixed_range": [2.0, 8.0]
        },
        {
            "id": "ghi_dia_05_may",
            "tif": "cordoba_ghi_diario_05_mayo_2km.tif",
            "name": "Mayo — GHI Medio Diario",
            "category": "Ciclo Estacional Mensual (kWh/m²/día)",
            "unit": "kWh/m²/día",
            "cmap": "inferno",
            "desc": "Media diaria mensual de irradiancia global horizontal en Mayo.",
            "format": ".2f",
            "fixed_range": [2.0, 8.0]
        },
        {
            "id": "ghi_dia_06_jun",
            "tif": "cordoba_ghi_diario_06_junio_2km.tif",
            "name": "Junio — GHI Medio Diario",
            "category": "Ciclo Estacional Mensual (kWh/m²/día)",
            "unit": "kWh/m²/día",
            "cmap": "inferno",
            "desc": "Media diaria mensual de irradiancia global horizontal en Junio (Mínimo invernal).",
            "format": ".2f",
            "fixed_range": [2.0, 8.0]
        },
        {
            "id": "ghi_dia_07_jul",
            "tif": "cordoba_ghi_diario_07_julio_2km.tif",
            "name": "Julio — GHI Medio Diario",
            "category": "Ciclo Estacional Mensual (kWh/m²/día)",
            "unit": "kWh/m²/día",
            "cmap": "inferno",
            "desc": "Media diaria mensual de irradiancia global horizontal en Julio (Inicio de recuperación).",
            "format": ".2f",
            "fixed_range": [2.0, 8.0]
        },
        # --- DIAGNÓSTICOS DE ESTABILIDAD ---
        {
            "id": "cv_interanual",
            "tif": "cordoba_cv_interanual_pct_2km.tif",
            "name": "Variabilidad Interanual (CV %)",
            "category": "Diagnósticos de Estabilidad Territorial",
            "unit": "%",
            "cmap": "YlOrRd",
            "desc": "Coeficiente de variación interanual porcentual observado entre 2020, 2021 y 2023.",
            "format": ".2f"
        },
        {
            "id": "delta_2",
            "tif": "cordoba_delta_2_pct_2km.tif",
            "name": "Sensibilidad Temporal Delta 2",
            "category": "Diagnósticos de Estabilidad Territorial",
            "unit": "%",
            "cmap": "RdBu_r",
            "desc": "Tasa de variación espacial relativa al incorporar 2 años respecto de 1 año (M20-21 vs M20).",
            "format": ".2f",
            "fixed_range": [-6.0, 6.0]
        },
        {
            "id": "delta_3",
            "tif": "cordoba_delta_3_pct_2km.tif",
            "name": "Sensibilidad Temporal Delta 3",
            "category": "Diagnósticos de Estabilidad Territorial",
            "unit": "%",
            "cmap": "RdBu_r",
            "desc": "Tasa de variación espacial relativa al incorporar 3 años respecto de 2 años (M20-21-23 vs M20-21).",
            "format": ".2f",
            "fixed_range": [-6.0, 6.0]
        }
    ]

    manifest = {
        "version": "1.0",
        "title": "Atlas Solar de la Provincia de Córdoba — Visor GeoTIFF 2 km",
        "convenio": "ATLAS II — FCEFyN / UNC & OHMC",
        "resolution": "2 km (0.018°)",
        "bounds": leaflet_bounds,
        "grid_shape": [ny, nx],
        "total_provincial_cells": 48645,
        "layers": []
    }

    print("\n🎨 4. Procesando capas y generando overlays RGBA + datos para el inspector...")

    for l_cfg in layers_config:
        tif_path = GEOTIFF_DIR / l_cfg['tif']
        if not tif_path.exists():
            print(f"   ⚠️ No encontrado: {tif_path.name}, salteando.")
            continue

        with rasterio.open(tif_path) as src:
            data = src.read(1)
            nodata = src.nodata

        valid_mask = (data != nodata) & (~np.isnan(data))
        valid_vals = data[valid_mask]

        if len(valid_vals) == 0:
            print(f"   ⚠️ Sin datos válidos en {l_cfg['id']}")
            continue

        # Estadísticas provinciales
        v_min = float(valid_vals.min())
        v_max = float(valid_vals.max())
        v_mean = float(valid_vals.mean())
        v_std = float(valid_vals.std())
        v_p10 = float(np.percentile(valid_vals, 10))
        v_p50 = float(np.percentile(valid_vals, 50))
        v_p90 = float(np.percentile(valid_vals, 90))

        # Estadísticas regionales (k=3)
        reg_stats = {}
        for r_id, r_name in [(0, "Sierras Altas"), (1, "Sierras Bajas / Trans."), (2, "Llanura Pampeana")]:
            m_r = valid_mask & (reg_data == r_id)
            if m_r.sum() > 0:
                vals_r = data[m_r]
                reg_stats[f"R{r_id}"] = {
                    "name": r_name,
                    "cells": int(m_r.sum()),
                    "mean": float(vals_r.mean()),
                    "std": float(vals_r.std())
                }

        # Rango para normalización cromática
        if "fixed_range" in l_cfg:
            norm_min, norm_max = l_cfg["fixed_range"]
        elif l_cfg.get("is_categorical", False):
            norm_min, norm_max = 0, 2
        else:
            norm_min = float(np.percentile(valid_vals, 0.5))
            norm_max = float(np.percentile(valid_vals, 99.5))

        # Generar imagen RGBA
        cmap_name = l_cfg['cmap']
        if l_cfg.get("is_categorical", False):
            # Asignación discreta para Regiones k=3
            # R0: Naranja (#E67E22), R1: Verde (#27AE60), R2: Azul (#2980B9)
            rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
            cat_colors = {
                0: [230, 126, 34, 230],
                1: [39, 174, 96, 230],
                2: [41, 128, 185, 230]
            }
            for c_val, col in cat_colors.items():
                m_c = valid_mask & (data == c_val)
                rgba[m_c] = col
            legend_colors = ["#E67E22", "#27AE60", "#2980B9"]
            legend_labels = ["Región 0 (Sierras Altas)", "Región 1 (Sierras Bajas)", "Región 2 (Llanura Pampeana)"]
        else:
            norm_val = np.clip((data - norm_min) / (norm_max - norm_min + 1e-9), 0.0, 1.0)
            cmap = plt.get_cmap(cmap_name)
            rgba_float = cmap(norm_val)
            rgba = (rgba_float * 255).astype(np.uint8)
            # Transparente fuera del polígono provincial
            rgba[~valid_mask] = [0, 0, 0, 0]
            legend_colors = get_hex_colors_from_cmap(cmap_name, n_stops=5)
            legend_labels = [
                f"{norm_min:{l_cfg['format']}}",
                f"{(norm_min*0.75 + norm_max*0.25):{l_cfg['format']}}",
                f"{(norm_min*0.50 + norm_max*0.50):{l_cfg['format']}}",
                f"{(norm_min*0.25 + norm_max*0.75):{l_cfg['format']}}",
                f"{norm_max:{l_cfg['format']}}"
            ]

        # Guardar imagen PNG en overlays/
        png_filename = f"{l_cfg['id']}.png"
        png_path = OVERLAYS_DIR / png_filename
        img = Image.fromarray(rgba, mode="RGBA")
        img.save(png_path, "PNG", optimize=True)

        # Histograma (20 bins)
        counts, bin_edges = np.histogram(valid_vals, bins=20)
        hist_data = {
            "counts": [int(c) for c in counts],
            "bins": [float(b) for b in bin_edges]
        }

        # Guardar matriz de valores redondeados para consulta instantánea (inspector)
        # Reemplazar nodata por null
        grid_clean = np.where(valid_mask, np.round(data, 2), -9999.0)
        grid_json_path = GRIDS_OUT_DIR / f"{l_cfg['id']}.json"
        with open(grid_json_path, "w", encoding="utf-8") as f_grid:
            json.dump(grid_clean.tolist(), f_grid)

        # Info para manifest
        layer_meta = {
            "id": l_cfg["id"],
            "name": l_cfg["name"],
            "category": l_cfg["category"],
            "unit": l_cfg["unit"],
            "desc": l_cfg["desc"],
            "format": l_cfg["format"],
            "tif_file": l_cfg["tif"],
            "tif_url": f"data/ready/maps_geotiff/{l_cfg['tif']}",
            "png_overlay": f"overlays/{png_filename}",
            "grid_data_url": f"data/grids/{l_cfg['id']}.json",
            "stats": {
                "min": v_min,
                "max": v_max,
                "mean": v_mean,
                "std": v_std,
                "p10": v_p10,
                "p50": v_p50,
                "p90": v_p90,
                "regional": reg_stats
            },
            "legend": {
                "colors": legend_colors,
                "labels": legend_labels,
                "norm_min": norm_min,
                "norm_max": norm_max,
                "is_categorical": l_cfg.get("is_categorical", False)
            },
            "histogram": hist_data
        }
        manifest["layers"].append(layer_meta)
        print(f"   ✓ Capa '{l_cfg['name']}': Media={v_mean:{l_cfg['format']}} {l_cfg['unit']} -> {png_filename}")

    # Guardar manifest
    manifest_path = DATA_OUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f_man:
        json.dump(manifest, f_man, indent=2)

    print(f"\n📄 Manifest general exportado a {manifest_path} ({len(manifest['layers'])} capas procesadas)")
    print("=" * 80)
    print("🎉 ¡TODOS LOS DATOS Y OVERLAYS DEL VISOR ESTÁN LISTOS!")
    print("=" * 80)


if __name__ == "__main__":
    main()
