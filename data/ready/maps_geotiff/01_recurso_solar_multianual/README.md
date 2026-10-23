# 01 — Recurso Solar Multianual (2020–2025)

Contiene los mapas ráster en formato Cloud-Optimized GeoTIFF (COG) de los componentes fundamentales de la radiación solar incidente en la Provincia de Córdoba, promediados sobre el sexenio 2020–2025.

## Archivos y Especificaciones

| Archivo | Variable | Unidad | Rango Prov. Típico | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| `cordoba_ghi_anual_medio_2020_2025_2km_cog.tif` | GHI | kWh/m²/año | 1.082 – 2.055 | Irradiancia Global Horizontal Anual Media (Media Prov: 1.889,3 kWh/m²/año) |
| `cordoba_dni_anual_medio_2020_2025_2km_cog.tif` | DNI | kWh/m²/año | 397 – 2.279 | Irradiancia Directa Normal Anual Media (Media Prov: 1.985,2 kWh/m²/año) |
| `cordoba_dhi_anual_medio_2020_2025_2km_cog.tif` | DHI | kWh/m²/año | 565 – 910 | Irradiancia Difusa Horizontal Anual Media (Media Prov: 604,9 kWh/m²/año) |
| `cordoba_psh_anual_medio_2020_2025_2km_cog.tif` | PSH / HSP | HSP/día | 2,96 – 5,63 | Horas Sol Pico promedio diario equivalente (Media Prov: 5,17 HSP/día) |

## Parámetros Técnicos Comunes
- **Sistema de Coordenadas:** EPSG:4326 (WGS 84).
- **Resolución Espacial:** 0.018° x 0.018° (~2,0 km).
- **Valor NoData:** `-9999.0`.
- **Estructura Interna:** Cloud-Optimized GeoTIFF (COG), Tiles 128x128, Overviews 2, 4, 8, Compresión DEFLATE.
