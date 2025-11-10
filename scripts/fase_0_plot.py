#!/usr/bin/env python3
import geopandas as gpd # type: ignore
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
from pathlib import Path

# Rutas
data_dir = Path("data")
grid_file = data_dir / "grids" / "cordoba_grid_5km.geojson"
boundary_file = data_dir / "cordoba_boundary.geojson"

# Cargar datos
cordoba = gpd.read_file(boundary_file)
grid = gpd.read_file(grid_file)

# Crear figura
fig, ax = plt.subplots(figsize=(8, 10))
cordoba.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)
grid.plot(ax=ax, color="red", markersize=3, alpha=0.5)

# Estilo
ax.set_title("Provincia de Córdoba – Rejilla base 5 km", fontsize=14)
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.grid(True, linestyle="--", alpha=0.5)

# Guardar
output = data_dir / "cordoba_grid_preview.png"
plt.tight_layout()
plt.savefig(output, dpi=300)
plt.show()

print(f"✅ Mapa guardado en {output}")
