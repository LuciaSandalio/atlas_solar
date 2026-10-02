#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generación de la Grilla Oficial de 2 km (0.018°) para el Atlas Solar de Córdoba.
Cumple con las especificaciones del Informe 1 (Sección 3.2):
- Paso de malla: 0.018° (~2.0 km)
- Cobertura: Provincia de Córdoba con buffer de 0.2° para evitar efectos de borde.
- Exporta GeoJSON y vista previa cartográfica.
"""

import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

def main():
    base_dir = Path(".")
    data_dir = base_dir / "data"
    grids_dir = data_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)
    
    bnd_file = data_dir / "cordoba_boundary.geojson"
    buf_file = data_dir / "cordoba_boundary_buffer.geojson"
    out_grid_file = grids_dir / "cordoba_grid_2km.geojson"
    out_png_file = grids_dir / "cordoba_grid_2km_preview.png"
    out_meta_file = base_dir / "metadata" / "run_manifest_2km.json"
    
    print("📍 Cargando límites provinciales...")
    with open(bnd_file, "r", encoding="utf-8") as f:
        bnd_json = json.load(f)
    with open(buf_file, "r", encoding="utf-8") as f:
        buf_json = json.load(f)
        
    bnd_coords = np.array(bnd_json["features"][0]["geometry"]["coordinates"][0])
    buf_coords = np.array(buf_json["features"][0]["geometry"]["coordinates"][0])
    
    path_bnd = MplPath(bnd_coords)
    path_buf = MplPath(buf_coords)
    
    # Especificación de 2 km (0.018°) según Informe 1
    step_deg = 0.018
    min_lon, max_lon = buf_coords[:, 0].min(), buf_coords[:, 0].max()
    min_lat, max_lat = buf_coords[:, 1].min(), buf_coords[:, 1].max()
    
    print(f"📐 Bounding box: Lon [{min_lon:.4f}, {max_lon:.4f}], Lat [{min_lat:.4f}, {max_lat:.4f}]")
    print(f"⚙️  Paso de malla: {step_deg:.4f}° (~2.0 km)")
    
    lons = np.arange(min_lon + step_deg / 2.0, max_lon, step_deg)
    lats = np.arange(min_lat + step_deg / 2.0, max_lat, step_deg)
    
    grid_x, grid_y = np.meshgrid(lons, lats)
    all_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    total_bbox_pts = len(all_points)
    
    print(f"📦 Total puntos en caja envolvente: {total_bbox_pts:,}")
    
    t0 = time.time()
    mask_buf = path_buf.contains_points(all_points)
    mask_bnd = path_bnd.contains_points(all_points)
    
    valid_points = all_points[mask_buf]
    in_prov = mask_bnd[mask_buf]
    t1 = time.time()
    
    n_total = len(valid_points)
    n_prov = int(in_prov.sum())
    n_buf_only = n_total - n_prov
    
    print(f"✅ Filtrado completado en {t1 - t0:.2f} s:")
    print(f"   • Total celdas de 2 km (con buffer 0.2°): {n_total:,}")
    print(f"   • Celdas estrictamente dentro de Córdoba: {n_prov:,}")
    print(f"   • Celdas en zona buffer externa:         {n_buf_only:,}")
    
    # Construir GeoJSON
    print(f"💾 Guardando GeoJSON en {out_grid_file}...")
    features = []
    for idx, (pt, in_p) in enumerate(zip(valid_points, in_prov)):
        feat = {
            "type": "Feature",
            "properties": {
                "cell_id": int(idx),
                "lon": float(round(pt[0], 6)),
                "lat": float(round(pt[1], 6)),
                "in_province": bool(in_p)
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(round(pt[0], 6)), float(round(pt[1], 6))]
            }
        }
        features.append(feat)
        
    geojson_out = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": features
    }
    
    with open(out_grid_file, "w", encoding="utf-8") as f:
        json.dump(geojson_out, f, indent=1)
        
    # Guardar Metadata
    meta = {
        "run_id": time.strftime("%Y%m%d_%H%M"),
        "region": "Córdoba",
        "grid_res_km": 2.0,
        "res_deg": step_deg,
        "n_cells_total": n_total,
        "n_cells_in_province": n_prov,
        "projection": "EPSG:4326",
        "buffer_deg": 0.2,
        "formato": "geojson",
        "fuente_datos": "GADM 4.1 / Convenio ATLAS II",
        "notas": "Grilla oficial de alta resolución 2 km (0.018°) para Informe 4"
    }
    with open(out_meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    # Generar Mapa Preview
    print(f"🎨 Generando mapa diagnóstico en {out_png_file}...")
    fig, ax = plt.subplots(figsize=(9, 11), dpi=200)
    
    # Buffer polygon
    ax.plot(buf_coords[:, 0], buf_coords[:, 1], color="#95A5A6", linestyle="--", linewidth=1.2, label="Buffer 0.2°")
    # Province boundary
    ax.plot(bnd_coords[:, 0], bnd_coords[:, 1], color="#2C3E50", linewidth=1.8, label="Límite Córdoba")
    
    # Puntos dentro de la provincia
    pts_prov = valid_points[in_prov]
    ax.scatter(pts_prov[:, 0], pts_prov[:, 1], s=0.8, color="#E67E22", alpha=0.6, label=f"Celdas Córdoba (n={n_prov:,})")
    
    # Puntos del buffer
    pts_buf = valid_points[~in_prov]
    ax.scatter(pts_buf[:, 0], pts_buf[:, 1], s=0.5, color="#BDC3C7", alpha=0.4, label=f"Celdas Buffer (n={n_buf_only:,})")
    
    ax.set_title("Atlas Solar Córdoba – Rejilla Oficial de 2 km (0.018°)\nBase Cartográfica para Informe 4", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Longitud (°O)", fontsize=11)
    ax.set_ylabel("Latitud (°S)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(out_png_file, dpi=200)
    plt.close()
    
    print(f"🎉 ¡Grilla de 2 km completada con éxito!")

if __name__ == "__main__":
    main()
