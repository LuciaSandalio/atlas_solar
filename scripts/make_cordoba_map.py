#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import fiona
import geopandas as gpd # Just for datasets path if needed, or avoid

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORDOBA_PATH = os.path.join(BASE_DIR, "data/cordoba_boundary.geojson")
STATIONS_PATH = os.path.join(BASE_DIR, "data/estaciones_sol.parquet")
OUT_FILE = os.path.join(BASE_DIR, "data/processed/cordoba_station_map.png")

# =========================
# Helpers
# =========================
def plot_polygon_collection(ax, geoms, **kwargs):
    """
    Plots a list of (exterior, interiors) geometries using matplotlib Path patches or Polygons.
    Simple version: just plot exteriors.
    geoms: list of list of (x,y) tuples, or list of polygons
    """
    patches = []
    for coords in geoms:
        # coords is list of (x,y)
        poly = MplPolygon(coords, closed=True, **kwargs)
        patches.append(poly)
    
    # Use PatchCollection for speed/style
    p = PatchCollection(patches, match_original=True, **kwargs)
    ax.add_collection(p)
    return p

def extract_polys_from_geometry(geom):
    """
    Extracts a list of exterior coordinates from a fiona/geojson geometry dict.
    Returns: list of [(x,y), ...]
    """
    polys = []
    if geom['type'] == 'Polygon':
        # coordinates is list of rings. First is exterior.
        ext = geom['coordinates'][0]
        polys.append(ext)
    elif geom['type'] == 'MultiPolygon':
        # coordinates is list of Polygons
        for poly_coords in geom['coordinates']:
            ext = poly_coords[0]
            polys.append(ext)
    return polys

def load_and_plot_fiona(ax, path, filter_func=None, **kwargs):
    """
    Loads features from path using fiona, filters them, and adds to ax.
    Returns the extent (minx, miny, maxx, maxy) of what was plotted.
    """
    print(f"Loading {path}...")
    minx, miny, maxx, maxy = 180, 90, -180, -90
    polys = []
    
    try:
        with fiona.open(path, 'r') as source:
            for feat in source:
                props = feat['properties']
                if filter_func and not filter_func(props):
                    continue
                
                geom = feat['geometry']
                if not geom: continue
                
                parts = extract_polys_from_geometry(geom)
                
                # Update bounds (rough)
                for part in parts:
                    # Force 2D
                    part_2d = [p[:2] for p in part]
                    polys.append(part_2d) # Add 2D part to the list for plotting
                    
                    xs = [p[0] for p in part_2d]
                    ys = [p[1] for p in part_2d]
                    minx = min(minx, min(xs))
                    maxx = max(maxx, max(xs))
                    miny = min(miny, min(ys))
                    maxy = max(maxy, max(ys))

    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

    # Plot
    patches = [MplPolygon(p, closed=True) for p in polys]
    
    # We apply kwargs to the collection
    p = PatchCollection(patches, **kwargs)
    ax.add_collection(p)
    return minx, miny, maxx, maxy

# =========================
# Main
# =========================
def main():
    print("Starting robust map generation...")
    
    # 1. Stations Data
    try:
        df = pd.read_parquet(STATIONS_PATH)
        print("Stations loaded.")
    except Exception as e:
        print(f"Error loading stations: {e}")
        return

    # Synthesize Type
    LAT_COL = "latitud"
    LON_COL = "longitud"
    NAME_COL = "Estación"
    TYPE_COL = "sensor"

    def infer_type(name):
        name_upper = str(name).upper()
        if "PILAR" in name_upper: return "CMP21"
        if "SMN" in name_upper: return "SMN"
        if "DAVIS" in name_upper: return "DAVIS"
        if "OMIXOM" in name_upper: return "OMIXOM"
        if "CMP6" in name_upper: return "CMP6"
        if "APRHI" in name_upper: return "APRHi" 
        return "Unknown"

    df[TYPE_COL] = df[NAME_COL].apply(infer_type)

    TYPE_STYLE = {
        "CMP21": dict(marker="s", s=60, color="tab:blue"),
        "CMP6":  dict(marker="o", s=45, color="tab:blue"),
        "DAVIS": dict(marker="^", s=45, color="tab:orange"),
        "SMN":   dict(marker="D", s=45, color="tab:green"),
        "OMIXOM":dict(marker="o", s=45, color="tab:red"),
        "APRHi": dict(marker="p", s=40, color="purple"),
        "Unknown": dict(marker=".", s=20, color="gray")
    }
    
    # 2. Setup Figure
    fig = plt.figure(figsize=(11, 4.5), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.12)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # 3. Panel A: South America
    # Use local shapefile for South America
    sa_path = os.path.join(BASE_DIR, "data/shapefiles/south_america/Paises_America_del_Sur_GEOGPSPERU_SuyoPomalia.shp")
    arg_path = os.path.join(BASE_DIR, "data/shapefiles/argentina/ARGENTINA PARTE CONTINENTAL AMERICANA ArcGIS 2025/shapes/Provincias.shp")
    
    # Plot South America Base
    if os.path.exists(sa_path):
        load_and_plot_fiona(ax1, sa_path, 
                            facecolor="#efe8b5", edgecolor="black", linewidth=0.4)
    else:
        print(f"Warning: {sa_path} not found.")

    # Plot Argentina Provinces on top (Highlight) - REMOVED per user request
    # if os.path.exists(arg_path):
    #     # Slightly darker or different fill to assist context
    #     load_and_plot_fiona(ax1, arg_path,
    #                         facecolor="#e0d69e", edgecolor="gray", linewidth=0.2)
    # else:
    #     print(f"Warning: {arg_path} not found.")

    ax1.set_xlim(-85, -30)
    ax1.set_ylim(-60, 15)
    ax1.set_title("South America (Córdoba highlighted)", fontsize=10)
    ax1.set_aspect("auto") # Item 1
    ax1.grid(True, linewidth=0.3, alpha=0.4)

    # 4. Panel B: Cordoba
    # Restore South America background (User request)
    if os.path.exists(sa_path):
        load_and_plot_fiona(ax2, sa_path,
                            facecolor="#efe8b5", edgecolor="black", linewidth=0.3)
    
    # Plot Cordoba Boundary
    # Item 2: Fill subtle beige, Thinner border
    bounds = load_and_plot_fiona(ax2, CORDOBA_PATH,
                                 facecolor="#efe2a8", edgecolor="black", linewidth=1.0, alpha=0.9)
    
    if bounds:
        minx, miny, maxx, maxy = bounds
        pad = 0.4 # Item 3: Tighter padding
        ax2.set_xlim(minx - pad, maxx + pad)
        ax2.set_ylim(miny - pad, maxy + pad)
        
        # Add box to Panel A
        from matplotlib.patches import Rectangle
        # Item 4: Adjusted box
        width = (maxx - minx) + 2*pad
        height = (maxy - miny) + 2*pad
        rect = Rectangle(
            (minx - pad, miny - pad),
            width,
            height,
            linewidth=2,
            edgecolor='#00ff00',
            facecolor='none',
            linestyle='--'
        )
        ax1.add_patch(rect)


    # 5. Plot Stations (Unified Style)
    # Item 4: Refined style
    UNIFIED_STYLE = dict(marker="o", s=45, color="#e41a1c")
    
    ax2.scatter(
        df[LON_COL],
        df[LAT_COL],
        label="Estaciones",
        edgecolors="black",
        linewidths=0.3, # Item 4
        zorder=5,
        **UNIFIED_STYLE
    )

    # Labels removed as per previous request
    
    ax2.set_title("Province of Córdoba – Solar radiation stations", fontsize=10)
    ax2.set_aspect("equal", adjustable="box") # Item 1
    # Item 3: Remove legend
    # ax2.legend(...) 
    
    # Item 1: Remove grid in Panel B
    ax2.grid(False)

    # Save
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Map saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
