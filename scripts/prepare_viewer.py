#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preparación Integral de Datos, Overlays y Base de Celdas para el Visor Web Interactivo
======================================================================================
Convenio ATLAS II — FCEFyN / UNC & OHMC — Gobierno de la Provincia de Córdoba
Periodo Definitivo: Atlas Solar Multianual 2020–2025 (Grilla Canónica 2 km)

Este script sincroniza de forma unificada:
1. Generación de overlays PNG de alta resolución en 'viewer/overlays/' con colormaps rigurosos,
   incluyendo la paleta oficial de Regiones Solares (R0 Azul, R1 Naranja, R2 Verde).
2. Generación de 'viewer/data/cell_database.js' indexado con floor para consulta O(1) exacta
   al clic, cruzando espacialmente con los 26 departamentos oficiales de Córdoba.
3. Generación de matrices JSON para el inspector de píxel en 'viewer/data/grids/'.
4. Exportación de 'viewer/data/manifest.json' con metadatos, histogramas y rangos reales 2020–2025.
"""

import sys
import os
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
READY_DIR = DATA_DIR / "ready"
DEF_DIR = READY_DIR / "atlas_definitivo_2km"

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
        stops.append(mcolors.to_hex(rgba))
    return stops


def main():
    print("=" * 80)
    print("🚀 PREPARANDO ASSETS Y DATOS DEL VISOR (ATLAS SOLAR MULTIANUAL 2020–2025)")
    print("=" * 80)

    # 1. Cargar datos maestros canónicos 2020–2025
    master_file = DEF_DIR / "atlas_anual_2020_2025_medio.parquet"
    conv_file = DEF_DIR / "atlas_convergencia_temporal_6years.parquet"
    
    if not master_file.exists():
        raise FileNotFoundError(f"Falta archivo maestro: {master_file}")
    if not conv_file.exists():
        raise FileNotFoundError(f"Falta archivo de convergencia: {conv_file}")

    print(f"📂 Cargando Parquet multianual maestro: {master_file.name}")
    df_multi = pd.read_parquet(master_file)
    df_conv = pd.read_parquet(conv_file)

    # Inyectar variables de convergencia y PSH
    df_multi['psh_anual_hsp_dia'] = df_multi['ghi_anual_kwh_m2'] / 365.25
    df_multi['cv_interanual_pct'] = df_conv['cv_interanual_pct']
    df_multi['delta_2_pct'] = df_conv['delta_2_pct']
    df_multi['delta_3_pct'] = df_conv['delta_3_pct']
    df_multi['delta_6_pct'] = df_conv['delta_6_pct']

    prov_mask = df_multi['in_province'].values
    n_prov = int(prov_mask.sum())
    print(f"   • Celdas totales: {len(df_multi):,} | Celdas provinciales interiores: {n_prov:,}")

    # 2. Configurar la geometría ráster regular 2D
    lons = np.sort(df_multi['lon'].unique())
    lats = np.sort(df_multi['lat'].unique())
    d_lon = 0.0180
    d_lat = 0.0180
    nx = len(lons)
    ny = len(lats)
    north = lats.max() + d_lat / 2.0
    west = lons.min() - d_lon / 2.0

    row_idx = np.floor((north - df_multi['lat'].values) / d_lat).astype(int)
    col_idx = np.floor((df_multi['lon'].values - west) / d_lon).astype(int)

    # Matriz 2D de máscara válida provincial
    valid_mask_2d = np.zeros((ny, nx), dtype=bool)
    valid_mask_2d[row_idx[prov_mask], col_idx[prov_mask]] = True

    # Matriz 2D de regiones solares k=3
    reg_2d = np.full((ny, nx), -1, dtype=int)
    reg_2d[row_idx[prov_mask], col_idx[prov_mask]] = df_multi.loc[prov_mask, 'region'].values.astype(int)

    # 3. Exportar capas vectoriales geojson simplificadas
    print("\n📦 Exportando capas vectoriales de departamentos y límite provincial...")
    cba_deptos_shp = ROOT_DIR / "shapes_cba" / "cba_deptos.shp"
    if cba_deptos_shp.exists():
        gdf_deptos = gpd.read_file(cba_deptos_shp).to_crs("EPSG:4326")
        gdf_deptos = gdf_deptos[['DEPARTAMTO', 'CABECERA', 'geometry']].copy()
        gdf_deptos['geometry'] = gdf_deptos['geometry'].simplify(0.002)
        deptos_json_path = DATA_OUT_DIR / "cordoba_deptos.geojson"
        gdf_deptos.to_file(deptos_json_path, driver="GeoJSON")
        print(f"   ✓ Departamentos exportados a {deptos_json_path.name}")

    boundary_geojson = DATA_DIR / "cordoba_boundary.geojson"
    if boundary_geojson.exists():
        gdf_bound = gpd.read_file(boundary_geojson).to_crs("EPSG:4326")
        bound_json_path = DATA_OUT_DIR / "cordoba_boundary.geojson"
        gdf_bound.to_file(bound_json_path, driver="GeoJSON")
        print(f"   ✓ Límite provincial exportado a {bound_json_path.name}")

    # 4. Generación de Base de Datos de Celdas (viewer/data/cell_database.js)
    print("\n💾 Generando base de datos de celdas 'cell_database.js' para consulta instantánea al clic...")
    df_prov = df_multi[prov_mask].copy()
    row_prov = row_idx[prov_mask]
    col_prov = col_idx[prov_mask]

    deptos = gpd.read_file(cba_deptos_shp).to_crs("EPSG:4326")
    gdf_cells = gpd.GeoDataFrame(df_prov, geometry=gpd.points_from_xy(df_prov['lon'], df_prov['lat']), crs="EPSG:4326")
    joined = gpd.sjoin(gdf_cells, deptos[['DEPARTAMTO', 'geometry']], how='left', predicate='within')
    joined['DEPARTAMTO'] = joined['DEPARTAMTO'].fillna('Córdoba')

    cell_db = {}
    for i in range(len(df_prov)):
        r = int(row_prov[i])
        c = int(col_prov[i])
        row_data = joined.iloc[i]
        
        ghi = round(float(row_data['ghi_anual_kwh_m2']), 1)
        dni = round(float(row_data['dni_anual_kwh_m2']), 1)
        dhi = round(float(row_data['dhi_anual_kwh_m2']), 1)
        psh = round(float(row_data['psh_anual_hsp_dia']), 2)
        poa_fix = round(float(row_data['poa_fijo_anual_kwh_m2']), 1)
        poa_trk = round(float(row_data['poa_tracker_anual_kwh_m2']), 1)
        yf_fix = round(float(row_data['yf_fixed_kwh_kwp']), 1)
        yf_trk = round(float(row_data['yf_tracker_kwh_kwp']), 1)
        cf_fix = round(float(row_data['cf_fixed_pct']), 1)
        cf_trk = round(float(row_data['cf_tracker_pct']), 1)
        gain_trk = round(float(row_data['ganancia_tracker_pct']), 2)
        reg = int(row_data['region'])
        dep = str(row_data['DEPARTAMTO'])
        
        cell_db[f"{r}_{c}"] = [ghi, dni, dhi, psh, poa_fix, poa_trk, yf_fix, yf_trk, cf_fix, cf_trk, gain_trk, reg, dep]

    cell_db_path = DATA_OUT_DIR / "cell_database.js"
    with open(cell_db_path, "w", encoding="utf-8") as f_db:
        f_db.write("window.ATLAS_CELL_DB = ")
        json.dump(cell_db, f_db, separators=(',', ':'))
        f_db.write(";\n")
    print(f"   ✓ 'cell_database.js' generado: {len(cell_db):,} celdas indexadas ({cell_db_path.stat().st_size / 1e6:.2f} MB)")

    # 5. Definición de Capas y Generación de Overlays PNG
    print("\n🎨 Generando overlays PNG para el visor interactivo...")

    layers_def = [
        # --- RECURSO SOLAR MULTIANUAL ---
        {
            "id": "ghi",
            "aliases": ["ghi_2020_2025", "ghi_quinquenal"],
            "name": "Irradiación Global Horizontal (GHI)",
            "category": "Recursos Solares Multianuales (2020–2025)",
            "unit": "kWh/m²·año",
            "values": df_multi['ghi_anual_kwh_m2'].values,
            "cmap": "inferno",
            "fixed_range": (1650.0, 2055.0),
            "format": ".0f",
            "desc": "Irradiación solar global incidente acumulada sobre plano horizontal en base anual."
        },
        {
            "id": "dni",
            "aliases": ["dni_2020_2025", "dni_quinquenal"],
            "name": "Irradiación Directa Normal (DNI)",
            "category": "Recursos Solares Multianuales (2020–2025)",
            "unit": "kWh/m²·año",
            "values": df_multi['dni_anual_kwh_m2'].values,
            "cmap": "inferno",
            "fixed_range": (1400.0, 2280.0),
            "format": ".0f",
            "desc": "Energía solar directa perpendicular al haz; recurso crítico para seguidores y concentración solar."
        },
        {
            "id": "dhi",
            "aliases": ["dhi_2020_2025", "dhi_quinquenal"],
            "name": "Irradiación Difusa Horizontal (DHI)",
            "category": "Recursos Solares Multianuales (2020–2025)",
            "unit": "kWh/m²·año",
            "values": df_multi['dhi_anual_kwh_m2'].values,
            "cmap": "cividis",
            "fixed_range": (565.0, 910.0),
            "format": ".0f",
            "desc": "Irradiación solar difusa dispersada por la atmósfera y cobertura nubosa (Media Prov: 604,9 kWh/m²·a)."
        },
        {
            "id": "psh",
            "aliases": ["psh_2020_2025", "psh_quinquenal"],
            "name": "Horas Sol Pico Diarias (HSP)",
            "category": "Recursos Solares Multianuales (2020–2025)",
            "unit": "HSP/día",
            "values": df_multi['psh_anual_hsp_dia'].values,
            "cmap": "inferno",
            "fixed_range": (4.50, 5.65),
            "format": ".2f",
            "desc": "Horas sol pico equivalentes diarias a 1000 W/m² (HSP/día = GHI anual / 365.25)."
        },
        # --- POTENCIAL FOTOVOLTAICO ---
        {
            "id": "poa_fijo",
            "aliases": ["poa_fijo_2020_2025", "poa_fijo_quinquenal"],
            "name": "POA Fijo — Inclinación 30° Norte",
            "category": "Potencial Fotovoltaico (2020–2025)",
            "unit": "kWh/m²·año",
            "values": df_multi['poa_fijo_anual_kwh_m2'].values,
            "cmap": "plasma",
            "fixed_range": (1700.0, 2210.0),
            "format": ".0f",
            "desc": "Irradiación total incidente sobre plano inclinado fijo a 30° orientado al Norte ecuatorial."
        },
        {
            "id": "poa_trk",
            "aliases": ["poa_tracker_2020_2025", "poa_tracker_quinquenal"],
            "name": "POA Tracker — Seguidor 1-Eje Backtracking",
            "category": "Potencial Fotovoltaico (2020–2025)",
            "unit": "kWh/m²·año",
            "values": df_multi['poa_tracker_anual_kwh_m2'].values,
            "cmap": "plasma",
            "fixed_range": (2050.0, 2700.0),
            "format": ".0f",
            "desc": "Irradiación total incidente sobre seguidor a 1 eje horizontal N-S con algoritmo de backtracking."
        },
        {
            "id": "yf_fijo",
            "aliases": ["yf_fijo_2020_2025", "yf_fijo_quinquenal"],
            "name": "Rendimiento Específico Fijo (Yf)",
            "category": "Potencial Fotovoltaico (2020–2025)",
            "unit": "kWh/kWp·año",
            "values": df_multi['yf_fixed_kwh_kwp'].values,
            "cmap": "viridis",
            "fixed_range": (1450.0, 1795.0),
            "format": ".0f",
            "desc": "Generación eléctrica neta en corriente alterna inyectada a la red por kWp instalado en plano fijo."
        },
        {
            "id": "yf_trk",
            "aliases": ["yf_tracker_2020_2025", "yf_tracker_quinquenal"],
            "name": "Rendimiento Específico Seguidor (Yf)",
            "category": "Potencial Fotovoltaico (2020–2025)",
            "unit": "kWh/kWp·año",
            "values": df_multi['yf_tracker_kwh_kwp'].values,
            "cmap": "viridis",
            "fixed_range": (1750.0, 2170.0),
            "format": ".0f",
            "desc": "Generación eléctrica neta inyectada con sistema de seguimiento solar horizontal en 1 eje."
        },
        {
            "id": "cf_fijo",
            "aliases": ["cf_fijo_2020_2025", "cf_fijo_quinquenal"],
            "name": "Factor de Capacidad Fijo (CF)",
            "category": "Potencial Fotovoltaico (2020–2025)",
            "unit": "%",
            "values": df_multi['cf_fixed_pct'].values,
            "cmap": "viridis",
            "fixed_range": (16.5, 20.5),
            "format": ".1f",
            "desc": "Factor de planta anual equivalente para sistemas en plano fijo (CF = Yf / 8760 h)."
        },
        {
            "id": "cf_trk",
            "aliases": ["cf_tracker_2020_2025", "cf_tracker_quinquenal"],
            "name": "Factor de Capacidad Seguidor (CF)",
            "category": "Potencial Fotovoltaico (2020–2025)",
            "unit": "%",
            "values": df_multi['cf_tracker_pct'].values,
            "cmap": "viridis",
            "fixed_range": (20.0, 24.8),
            "format": ".1f",
            "desc": "Factor de planta anual para parques fotovoltaicos con seguidores solares en 1 eje."
        },
        {
            "id": "gain_trk",
            "aliases": ["ganancia_tracker_2020_2025", "ganancia_tracker_quinquenal"],
            "name": "Ganancia por Seguimiento Solar",
            "category": "Potencial Fotovoltaico (2020–2025)",
            "unit": "%",
            "values": df_multi['ganancia_tracker_pct'].values,
            "cmap": "RdYlGn",
            "fixed_range": (14.4, 23.4),
            "format": ".2f",
            "desc": "Incremento porcentual de generación eléctrica de seguidor a 1 eje respecto a estructura fija."
        },
        # --- REGIONALIZACIÓN SOLAR (PALETA OFICIAL INFORMES: R0 AZUL, R1 NARANJA, R2 VERDE) ---
        {
            "id": "regiones_k3",
            "aliases": ["regiones_k3"],
            "name": "Regiones Solares Homogéneas (k=3)",
            "category": "Geografía & Clasificación",
            "unit": "Zonas",
            "values": df_multi['region'].values.astype(float),
            "is_categorical": True,
            "desc": "Zonas solares: R0=Sierras Altas (Azul), R1=Transición/Piedemonte (Naranja), R2=Llanura Pampeana (Verde)."
        },
        # --- DIAGNÓSTICOS Y ESTABILIDAD ---
        {
            "id": "cv",
            "aliases": ["cv_interanual_2020_2025", "cv_interanual_quinquenal"],
            "name": "Variabilidad Interanual (CV %)",
            "category": "Diagnósticos & Estabilidad",
            "unit": "%",
            "values": df_multi['cv_interanual_pct'].values,
            "cmap": "YlOrRd",
            "fixed_range": (0.5, 4.5),
            "format": ".1f",
            "desc": "Coeficiente de variación interanual sexenal (Media Prov: 2,14%)."
        },
        {
            "id": "delta_2",
            "aliases": ["delta_2"],
            "name": "Sensibilidad Temporal Delta 2 (%)",
            "category": "Diagnósticos & Estabilidad",
            "unit": "%",
            "values": df_multi['delta_2_pct'].values,
            "cmap": "RdBu_r",
            "fixed_range": (-3.0, 3.0),
            "format": ".2f",
            "desc": "Variación porcentual acumulada al pasar de 1 a 2 años de análisis (2020 a 2021)."
        },
        {
            "id": "delta_3",
            "aliases": ["delta_3"],
            "name": "Sensibilidad Temporal Delta 3 (%)",
            "category": "Diagnósticos & Estabilidad",
            "unit": "%",
            "values": df_multi['delta_3_pct'].values,
            "cmap": "RdBu_r",
            "fixed_range": (-3.0, 3.0),
            "format": ".2f",
            "desc": "Variación porcentual acumulada al incorporar el tercer año (2022)."
        },
        {
            "id": "delta_6",
            "aliases": ["delta_6"],
            "name": "Sensibilidad Temporal Delta 6 (%)",
            "category": "Diagnósticos & Estabilidad",
            "unit": "%",
            "values": df_multi['delta_6_pct'].values,
            "cmap": "RdBu_r",
            "fixed_range": (-1.5, 1.5),
            "format": ".2f",
            "desc": "Variación porcentual acumulada al incorporar el año 2025 (|Delta_6| media = 0,15%)."
        }
    ]

    manifest = {
        "title": "Atlas Solar Multianual de Córdoba (2020–2025)",
        "period": "2020-2025",
        "resolution": "2.0 km (0.018 deg)",
        "crs": "EPSG:4326",
        "bounds": [
            [float(-35.198733), float(-65.97825)],
            [float(-29.294733), float(-61.58625)]
        ],
        "grid_size": [ny, nx],
        "total_cells": len(df_multi),
        "prov_cells": n_prov,
        "layers": []
    }

    for l_cfg in layers_def:
        raw_vals = l_cfg["values"]
        grid_2d = np.full((ny, nx), -9999.0, dtype=np.float32)
        grid_2d[row_idx[prov_mask], col_idx[prov_mask]] = raw_vals[prov_mask]

        valid_vals = raw_vals[prov_mask]
        v_min = float(valid_vals.min())
        v_max = float(valid_vals.max())
        v_mean = float(valid_vals.mean())
        v_std = float(valid_vals.std())
        v_p10 = float(np.percentile(valid_vals, 10))
        v_p50 = float(np.percentile(valid_vals, 50))
        v_p90 = float(np.percentile(valid_vals, 90))

        # Estadísticas zonales (k=3)
        reg_stats = {}
        reg_labels = [(0, "Sierras Altas (R0)"), (1, "Transición / Piedemonte (R1)"), (2, "Llanura Pampeana (R2)")]
        for r_id, r_name in reg_labels:
            m_r = (df_multi.loc[prov_mask, 'region'] == r_id)
            if m_r.sum() > 0:
                vals_r = valid_vals[m_r]
                reg_stats[f"R{r_id}"] = {
                    "name": r_name,
                    "cells": int(m_r.sum()),
                    "mean": float(vals_r.mean()),
                    "std": float(vals_r.std())
                }

        # Generar imagen RGBA
        if l_cfg.get("is_categorical", False):
            # PALETA OFICIAL RIGUROSA:
            # R0: Azul (#2980b9) -> RGBA [41, 128, 185, 235]
            # R1: Naranja (#e67e22) -> RGBA [230, 126, 34, 235]
            # R2: Verde (#27ae60) -> RGBA [39, 174, 96, 235]
            rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
            cat_colors = {
                0: [41, 128, 185, 235],   # Azul - R0 Sierras Altas
                1: [230, 126, 34, 235],   # Naranja - R1 Transición
                2: [39, 174, 96, 235]     # Verde - R2 Llanura
            }
            for c_val, col in cat_colors.items():
                m_c = valid_mask_2d & (reg_2d == c_val)
                rgba[m_c] = col
            legend_colors = ["#2980b9", "#e67e22", "#27ae60"]
            legend_labels = ["Región 0 (Sierras Altas)", "Región 1 (Transición / Piedemonte)", "Región 2 (Llanura Pampeana)"]
            norm_min, norm_max = 0, 2
        else:
            norm_min, norm_max = l_cfg["fixed_range"]
            norm_val = np.clip((grid_2d - norm_min) / (norm_max - norm_min + 1e-9), 0.0, 1.0)
            cmap = plt.get_cmap(l_cfg['cmap'])
            rgba_float = cmap(norm_val)
            rgba = (rgba_float * 255).astype(np.uint8)
            rgba[~valid_mask_2d] = [0, 0, 0, 0]
            legend_colors = get_hex_colors_from_cmap(l_cfg['cmap'], n_stops=5)
            fmt = l_cfg['format']
            legend_labels = [
                f"{norm_min:{fmt}}",
                f"{(norm_min*0.75 + norm_max*0.25):{fmt}}",
                f"{(norm_min*0.50 + norm_max*0.50):{fmt}}",
                f"{(norm_min*0.25 + norm_max*0.75):{fmt}}",
                f"{norm_max:{fmt}}"
            ]

        # Guardar archivo PNG principal y todos sus alias
        img = Image.fromarray(rgba, mode="RGBA")
        for alias in l_cfg["aliases"]:
            png_path = OVERLAYS_DIR / f"{alias}.png"
            img.save(png_path, "PNG", optimize=True)

        # Histograma (20 bins)
        counts, bin_edges = np.histogram(valid_vals, bins=20)
        hist_data = {
            "counts": [int(c) for c in counts],
            "bins": [float(b) for b in bin_edges]
        }

        # Guardar matriz JSON para el inspector rápido
        grid_clean = np.where(valid_mask_2d, np.round(grid_2d, 2), -9999.0)
        grid_json_path = GRIDS_OUT_DIR / f"{l_cfg['id']}.json"
        with open(grid_json_path, "w", encoding="utf-8") as f_grid:
            json.dump(grid_clean.tolist(), f_grid)

        layer_meta = {
            "id": l_cfg["id"],
            "name": l_cfg["name"],
            "category": l_cfg["category"],
            "unit": l_cfg["unit"],
            "desc": l_cfg["desc"],
            "format": l_cfg.get("format", ".1f"),
            "png_overlay": f"overlays/{l_cfg['aliases'][0]}.png",
            "grid_data_url": f"data/grids/{l_cfg['id']}.json",
            "stats": {
                "min": v_min, "max": v_max, "mean": v_mean, "std": v_std,
                "p10": v_p10, "p50": v_p50, "p90": v_p90,
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
        print(f"   ✓ Capa '{l_cfg['name']}': Media={v_mean:{l_cfg.get('format', '.1f')}} {l_cfg['unit']} -> {l_cfg['aliases']}")

    # Guardar manifest.json
    manifest_path = DATA_OUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f_man:
        json.dump(manifest, f_man, indent=2)
    print(f"\n📄 Manifest exportado a {manifest_path} ({len(manifest['layers'])} capas)")

    print("\n" + "=" * 80)
    print("✅ PREPARACIÓN DE ASSETS DEL VISOR COMPLETADA CON ÉXITO")
    print("=" * 80)


if __name__ == '__main__':
    main()
