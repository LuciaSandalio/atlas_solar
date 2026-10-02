# -*- coding: utf-8 -*-

# ========================================
# CODE CELL 1
# ========================================
# BLOQUE 1: Instalación de librerías necesarias
!pip install geopandas
!pip install scikit-learn
!pip install fiona

# BLOQUE 2: Importación de librerías
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
#from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import silhouette_samples
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


# ========================================
# CODE CELL 2
# ========================================
# BLOQUE 2: Montaje de Google Drive
#drive.mount('/content/drive')
import pandas as pd

# ========================================
# CODE CELL 3
# ========================================
# BLOQUE 3: Cargar datos
# CAMBIAR ESTA RUTA POR TU ARCHIVO
dict_wrf = pd.read_pickle('/home/ana_luz/remoto/resultados/est_T/dfs/dict_wrf.pkl')
dict_wrf.keys()
dict_wrf['Bouwer BioagroindustriaCba']

# print("Cargando archivo...")
# gdf = gpd.read_file(ruta_archivo)
# print(f"✓ Archivo cargado: {len(gdf)} registros, {len(gdf.columns)} columnas")


# ========================================
# CODE CELL 4
# ========================================
df_wrf = []

for key in dict_wrf.keys():
    df = dict_wrf[key].copy()
    
    df['estacion'] = key
    
    # asegurar datetime
    df.index = pd.to_datetime(df.index)
    
    # convertir solo columnas numéricas
    cols = df.columns.drop('estacion')
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    
    # promedio anual
    df_anual = df.resample('Y').mean(numeric_only=True)
    
    # agregar metadata
    df_anual['estacion'] = key
    df_anual['anio'] = df_anual.index.year
    
    # eliminar 2025
    df_anual = df_anual[df_anual['anio'] != 2025]
    
    df_wrf.append(df_anual)

# unir todo
df_wrf = pd.concat(df_wrf)

# promedio entre años por estación
df_final = df_wrf.groupby('estacion').mean(numeric_only=True).reset_index()

df_final

# ========================================
# CODE CELL 5
# ========================================
df_final.drop(columns=['anio'], inplace=True)

# ========================================
# CODE CELL 6
# ========================================
df_coords = pd.read_csv('/home/ana_luz/remoto/resultados/est_T/dfs/estaciones_lat_lon.csv')
df_coords.head()

# ========================================
# CODE CELL 7
# ========================================
import re
import unicodedata

def normalizar(texto):
    if pd.isna(texto):
        return texto
    
    texto = texto.lower()
    
    # eliminar acentos
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    
    # reemplazar TODO lo no alfanumérico por espacio
    texto = re.sub(r'[^a-z0-9]', ' ', texto)
    
    # eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

# ========================================
# CODE CELL 8
# ========================================
df_final['match'] = df_final['estacion'].apply(normalizar)
df_coords['match'] = df_coords['nombre'].apply(normalizar)

df_merge = df_final.merge(
    df_coords[['match', 'latitud', 'longitud']],
    on='match',
    how='left'
)

# ========================================
# CODE CELL 9
# ========================================
df_merge

# ========================================
# CODE CELL 10
# ========================================
gdf = gpd.GeoDataFrame(
    df_merge,
    geometry=gpd.points_from_xy(df_merge['longitud'], df_merge['latitud']),
    crs='EPSG:4326'
)

# ========================================
# CODE CELL 11
# ========================================
# BLOQUE 4: Selección inteligente de variables
# Excluir variables no útiles
variables_excluir = ['geometry', 'id', 'fid', 'id_grilla', 'CLUSTER_ID']
variables_usar = [col for col in gdf.columns if col not in variables_excluir]

# Solo variables numéricas
datos = gdf[variables_usar].select_dtypes(include=[np.number])
print(f"Variables iniciales: {len(datos.columns)}")
print(f"Variables: {list(datos.columns)}")

# LIMPIEZA DE CORRELACIONES con prioridad a MEDIA
print(f"\n🔍 Analizando correlaciones...")

# Calcular matriz de correlación
corr_matrix = datos.corr().abs()

# Función para decidir qué variable mantener entre dos correlacionadas
def decidir_variable_prioritaria(var1, var2):
    """
    Decide qué variable mantener basándose en prioridades:
    1. La que contenga 'media' en el nombre
      """
    # Convertir a minúsculas para comparación
    v1_lower = var1.lower()
    v2_lower = var2.lower()

    # Prioridad 1: Variables con 'media'
    v1_has_media = 'media' in v1_lower
    v2_has_media = 'media' in v2_lower

    if v1_has_media and not v2_has_media:
        return var1
    elif v2_has_media and not v1_has_media:
        return var2


# Encontrar pares de variables altamente correlacionadas (>0.90)
variables_a_eliminar = set()
correlaciones_encontradas = []

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        var1 = corr_matrix.columns[i]
        var2 = corr_matrix.columns[j]
        correlacion = corr_matrix.iloc[i, j]

        if correlacion > 0.90:  # Correlación alta
            # Decidir cuál mantener
            mantener = decidir_variable_prioritaria(var1, var2)
            eliminar = var2 if mantener == var1 else var1

            variables_a_eliminar.add(eliminar)
            correlaciones_encontradas.append({
                'eliminada': eliminar,
                'mantenida': mantener,
                'correlacion': correlacion
            })

# Mostrar resultados
if correlaciones_encontradas:
    print(f"Variables correlacionadas encontradas:")
    for item in correlaciones_encontradas:
        print(f"  • Eliminar '{item['eliminada']}' - correlacionada con '{item['mantenida']}' (r={item['correlacion']:.3f})")

    # Eliminar variables correlacionadas
    datos = datos.drop(columns=list(variables_a_eliminar))
    print(f"\n✂️ Eliminadas {len(variables_a_eliminar)} variables por correlación")
else:
    print("✓ No se encontraron correlaciones altas (>0.90)")

print(f"\nVariables finales: {len(datos.columns)}")
print(f"Variables: {list(datos.columns)}")


# ========================================
# CODE CELL 12
# ========================================
# BLOQUE 5: Limpieza básica
# Eliminar filas con valores nulos
datos_limpios = datos.dropna()
print(f"Registros válidos: {len(datos_limpios)}")

# Normalizar datos
scaler = StandardScaler()
datos_norm = scaler.fit_transform(datos_limpios)

# ========================================
# CODE CELL 13
# ========================================
# BLOQUE 6: Clustering K-means simple
# Probar solo 3 opciones de clusters para ser rápido
print("\nProbando diferentes números de clusters...")
resultados = {}

for k in [4,5,6,7,8,9,10]:  # Solo 3 opciones para ser rápido
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)  # Menos inicializaciones
    labels = kmeans.fit_predict(datos_norm)
    silhouette = silhouette_score(datos_norm, labels)
    resultados[k] = {'silhouette': silhouette, 'model': kmeans, 'labels': labels}
    print(f"K={k}: Silhouette = {silhouette:.3f}")

# Seleccionar el mejor
mejor_k = max(resultados.keys(), key=lambda k: resultados[k]['silhouette'])
mejor_score = resultados[mejor_k]['silhouette']
mejor_modelo = resultados[mejor_k]['model']
clusters_finales = resultados[mejor_k]['labels']

print(f"\n🎯 Mejor opción: {mejor_k} clusters (Silhouette: {mejor_score:.3f})")


# ========================================
# CODE CELL 14
# ========================================
# BLOQUE 7: Asignar clusters al GeoDataFrame
gdf_resultado = gdf.copy()
gdf_resultado['cluster'] = -1  # Valor por defecto
gdf_resultado.loc[datos_limpios.index, 'cluster'] = clusters_finales

print(f"\nDistribución de clusters:")
distribucion = gdf_resultado['cluster'].value_counts().sort_index()
for cluster, count in distribucion.items():
    if cluster == -1:
        print(f"  Sin asignar: {count}")
    else:
        print(f"  Cluster {cluster}: {count}")

# BLOQUE 8: Visualización simple
plt.figure(figsize=(12, 4))

# Gráfico 1: Distribución de clusters
plt.subplot(1, 3, 1)
clusters_validos = gdf_resultado[gdf_resultado['cluster'] != -1]['cluster']
plt.hist(clusters_validos, bins=mejor_k, alpha=0.7, color='skyblue', edgecolor='black')
plt.title(f'Distribución de {mejor_k} Clusters')
plt.xlabel('Cluster')
plt.ylabel('Frecuencia')

# Gráfico 2: Comparación de Silhouette scores
plt.subplot(1, 3, 2)
k_values = list(resultados.keys())
silhouette_values = [resultados[k]['silhouette'] for k in k_values]
plt.bar(k_values, silhouette_values, color='lightgreen', alpha=0.7)
plt.axhline(y=mejor_score, color='red', linestyle='--', alpha=0.7)
plt.title('Coeficientes de Silhouette')
plt.xlabel('Número de Clusters')
plt.ylabel('Silhouette Score')

# Gráfico 3: Mapa (si hay geometría)
plt.subplot(1, 3, 3)
if 'geometry' in gdf_resultado.columns and not gdf_resultado.geometry.isna().all():
    gdf_plot = gdf_resultado[gdf_resultado['cluster'] != -1]
    gdf_plot.plot(column='cluster', cmap='viridis', ax=plt.gca(), markersize=15)
    plt.title('Mapa de Clusters')
    plt.axis('off')
else:
    plt.text(0.5, 0.5, 'Sin geometría\ndisponible', ha='center', va='center',
             transform=plt.gca().transAxes, fontsize=12)
    plt.title('Mapa no disponible')

plt.tight_layout()
plt.show()

# BLOQUE 9: Estadísticas básicas por cluster
print(f"\n📊 Estadísticas por cluster (primeras 3 variables):")
primeras_vars = datos.columns  # Solo las primeras 3 para ser rápido

for cluster_id in sorted(gdf_resultado['cluster'].unique()):
    if cluster_id != -1:
        cluster_data = gdf_resultado[gdf_resultado['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} (n={len(cluster_data)}):")
        for var in primeras_vars:
            if var in cluster_data.columns:
                media = cluster_data[var].mean()
                std = cluster_data[var].std()
                print(f"  {var}: Media={media:.2f}, Std={std:.2f}")

# BLOQUE 10: Reproyección y exportación
print(f"\n💾 Preparando exportación...")

# Reproyectar a EPSG:22174
if gdf_resultado.crs is not None:
    gdf_final = gdf_resultado.to_crs(epsg=22174)
    print(f"✓ Reproyectado a EPSG:22174")
else:
    gdf_final = gdf_resultado
    print("⚠️ Sin CRS original, manteniendo coordenadas originales")

# Exportar
ruta_salida = '/home/ana_luz/remoto/resultados/est_T/dfs/cluster.gpkg'
try:
    gdf_final.to_file(ruta_salida, driver='GPKG')
    print(f"✓ Archivo guardado: {ruta_salida}")

    # Verificar
    verificacion = gpd.read_file(ruta_salida)
    print(f"✓ Verificación: {len(verificacion)} registros guardados")

except Exception as e:
    print(f"❌ Error al guardar: {e}")

# BLOQUE 11: Resumen final
print(f"\n{'='*50}")
print("RESUMEN FINAL")
print(f"{'='*50}")
print(f"📁 Archivo procesado: {len(gdf)} registros")
print(f"🔢 Variables utilizadas: {len(datos.columns)}")
print(f"✅ Registros válidos: {len(datos_limpios)}")
print(f"🎯 Clusters generados: {mejor_k}")
print(f"📊 Calidad (Silhouette): {mejor_score:.3f}")
print(f"💾 Archivo final: {ruta_salida}")

# Interpretación simple del Silhouette
if mejor_score > 0.5:
    calidad = "BUENO"
elif mejor_score > 0.25:
    calidad = "ACEPTABLE"
else:
    calidad = "POBRE - Considera revisar variables"

print(f"🏆 Calidad del clustering: {calidad}")
print(f"{'='*50}")

# ========================================
# CODE CELL 15
# ========================================
gdf_resultado

# ========================================
# CODE CELL 16
# ========================================
gdf_resultado = gdf_resultado.to_crs(epsg=4326)

gdf_resultado.explore(
    column="cluster",
    categorical=True,
    cmap="tab10"
)

# ========================================
# CODE CELL 17
# ========================================
# BLOQUE: análisis y visualización de clusters

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# 1. Filtrar datos válidos (sin cluster = -1)
df_valid = gdf_resultado[gdf_resultado['cluster'] != -1].copy()

# 2. Seleccionar SOLO variables usadas en clustering
vars_cluster = datos.columns.tolist()
df_num = df_valid[vars_cluster]

# 3. Normalización min-max
df_norm = (df_num - df_num.min()) / (df_num.max() - df_num.min())

# 4. Media por cluster (ya normalizado)
df_mean_norm = df_norm.groupby(df_valid['cluster']).mean()

# --------------------------------------------------
# 5. HEATMAP (recomendado)
# --------------------------------------------------

plt.figure(figsize=(10,6))
sns.heatmap(df_mean_norm, cmap='coolwarm', annot=False)
plt.title('Perfil normalizado de variables por cluster')
plt.xlabel('Variables')
plt.ylabel('Cluster')
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. RADAR PLOT (opcional)
# --------------------------------------------------

variables = df_mean_norm.columns
N = len(variables)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(7,7))

for i, row in df_mean_norm.iterrows():
    values = row.tolist()
    values += values[:1]
    plt.plot(angles, values, label=f'Cluster {i}')
    plt.fill(angles, values, alpha=0.1)

plt.xticks(angles[:-1], variables, rotation=90)
plt.title('Perfiles normalizados por cluster')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 7. BOXPLOTS (validación de separación)
# --------------------------------------------------

for var in vars_cluster:
    plt.figure()
    sns.boxplot(
        data=df_valid,
        x='cluster',
        y=var
    )
    plt.title(f'Distribución de {var} por cluster')
    plt.tight_layout()
    plt.show()

