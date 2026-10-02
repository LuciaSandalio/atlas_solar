from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os

df = pd.read_csv('data/raw/aws/resumen_estaciones_sol_todas.csv')
sel = df[df['station_id'].isin([115.0, 169.0, 274.0, 105.0])].copy()
deptos_shp = Path("shapes_cba/cba_deptos.shp")

fig, ax = plt.subplots(figsize=(8, 10))

# Mapa de Córdoba (departamentos)
cordoba = gpd.read_file('data/gadm/gadm41_ARG_2.shp')
deptos_gdf = gpd.read_file(deptos_shp) 
cordoba = cordoba[cordoba['NAME_1'] == 'Córdoba']
deptos_gdf.plot(ax=ax, facecolor="none", edgecolor="#95A5A6", linewidth=0.8, zorder=1)
cordoba.plot(ax=ax, facecolor="none", edgecolor='#333333', linewidth=0.5, alpha=0.9)

# Limpiar strings de lat y lon (tienen comas)
sel['lat'] = sel['lat'].astype(str).str.replace(',', '.').astype(float)
sel['lon'] = sel['lon'].astype(str).str.replace(',', '.').astype(float)

# Agregar la región manualmente para las etiquetas
regiones = {115.0: "Región 0", 169.0: "Región 1", 274.0: "Región 2", 105.0: "Región 3"}
sel['region'] = sel['station_id'].map(regiones)

# Plotear puntos
ax.scatter(sel['lon'], sel['lat'], color='red', s=100, edgecolor='black', zorder=5)

# Etiquetas
for i, row in sel.iterrows():
    label = f"{row['region']}\n{row['station_name']}"
    ax.annotate(label, (row['lon'], row['lat']), xytext=(5, 5), 
                textcoords='offset points', fontsize=8, weight='bold', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray', boxstyle='round,pad=0.3'))

ax.set_title("Estaciones Representativas Seleccionadas por Región", fontsize=14, weight='bold')
ax.axis('off')
plt.tight_layout()
os.makedirs('reportes/figures', exist_ok=True)
plt.savefig('reportes/figures/mapa_4_estaciones.png', dpi=150)
print("Saved to reportes/figures/mapa_4_estaciones.png")
