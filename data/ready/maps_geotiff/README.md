# Catálogo Cartográfico Oficial de Rásters GeoTIFF (COG)
## Atlas Solar de la Provincia de Córdoba (Convenio ATLAS II)

**Instituciones ejecutoras:** Facultad de Ciencias Exactas, Físicas y Naturales (FCEFyN — UNC) y Observatorio Hidro-Meteorológico de Córdoba (OHMC).

**Periodo Climatológico:** 2020–2025 (Sexenal, 6 años completos).
**Resolución Espacial:** 2,0 km × 2,0 km (0.018° × 0.018° angular, 48.645 celdas provinciales interiores).
**Sistema de Referencia Espacial:** EPSG:4326 (WGS 84).
**Formato de Almacenamiento:** Cloud-Optimized GeoTIFF (COG) con compresión interna DEFLATE sin pérdida, teselas (*tiles*) de 128×128 píxeles y pirámides multinivel (*overviews*) de 2, 4 y 8.
**Valor de No-Dato (*NoData*):** `-9999.0`.

---

## Organización de Carpetas Temáticas

Para facilitar la búsqueda y uso por parte de usuarios externos (organismos públicos, desarrolladores de proyectos fotovoltaicos, consultoras y la comunidad científica), los productos ráster están organizados en las siguientes carpetas estructuradas:

```
maps_geotiff/
│
├── 01_recurso_solar_multianual/          -> GHI, DNI, DHI anual medio y Horas Sol Pico (HSP)
├── 02_potencial_fotovoltaico_multianual/ -> POA Fijo 30°, POA Seguidor 1-eje, Yf, CF y Ganancia Tracker
├── 03_ciclo_mensual_multianual/         -> 12 capas mensuales (Enero a Diciembre) en kWh/m²/mes
├── 04_series_anuales_individuales/      -> Series completas año por año (2020 a 2025) en subcarpetas
├── 05_variabilidad_y_convergencia/      -> Coeficiente de variación CV (%) y sensibilidades Delta_2 a Delta_6
├── 06_regionalizacion_y_zonas_homogeneas/ -> Clasificación de zonas solares homogéneas (k=3)
└── 00_archivo_historico_previo/         -> Respaldo de versiones intermedias y preliminares históricas
```

## Resumen Rápido de Variables Principales (Climatología Sexenal 2020–2025)

| Variable | Unidad | Media Prov. | Rango Min – Max | Ruta Relativa |
| :--- | :--- | :--- | :--- | :--- |
| **GHI Anual** | kWh/m²/año | 1.889,3 | 1.081,9 – 2.054,7 | `01_recurso_solar_multianual/cordoba_ghi_anual_medio_2020_2025_2km_cog.tif` |
| **DNI Anual** | kWh/m²/año | 1.985,2 | 396,6 – 2.279,2 | `01_recurso_solar_multianual/cordoba_dni_anual_medio_2020_2025_2km_cog.tif` |
| **DHI Anual** | kWh/m²/año | 604,9 | 564,5 – 910,2 | `01_recurso_solar_multianual/cordoba_dhi_anual_medio_2020_2025_2km_cog.tif` |
| **HSP Diaria** | HSP/día | 5,17 | 2,96 – 5,63 | `01_recurso_solar_multianual/cordoba_psh_anual_medio_2020_2025_2km_cog.tif` |
| **POA Fijo 30°** | kWh/m²/año | 2.016,6 | 1.042,3 – 2.206,0 | `02_potencial_fotovoltaico_multianual/cordoba_poa_fijo_anual_medio_2020_2025_2km_cog.tif` |
| **POA Seguidor** | kWh/m²/año | 2.461,6 | 1.201,5 – 2.701,6 | `02_potencial_fotovoltaico_multianual/cordoba_poa_tracker_anual_medio_2020_2025_2km_cog.tif` |
| **$Y_f$ Fijo** | kWh/kWp/año | 1.644,7 | 870,5 – 1.792,6 | `02_potencial_fotovoltaico_multianual/cordoba_yf_fijo_anual_medio_2020_2025_2km_cog.tif` |
| **$Y_f$ Seguidor** | kWh/kWp/año | 1.982,2 | 995,6 – 2.167,5 | `02_potencial_fotovoltaico_multianual/cordoba_yf_tracker_anual_medio_2020_2025_2km_cog.tif` |
| **CF Fijo** | % | 18,77% | 9,94% – 20,46% | `02_potencial_fotovoltaico_multianual/cordoba_cf_fijo_anual_medio_2020_2025_2km_cog.tif` |
| **CF Seguidor** | % | 22,63% | 11,37% – 24,74% | `02_potencial_fotovoltaico_multianual/cordoba_cf_tracker_anual_medio_2020_2025_2km_cog.tif` |
| **Ganancia Seguidor** | % | +20,51% | 14,38% – 23,36% | `02_potencial_fotovoltaico_multianual/cordoba_ganancia_tracker_anual_medio_2020_2025_2km_cog.tif` |
| **CV Interanual** | % | 2,14% | 0,47% – 16,99% | `05_variabilidad_y_convergencia/cordoba_cv_interanual_2020_2025_2km_cog.tif` |

## Uso en Software SIG (QGIS, ArcGIS, Python, R)
Todos los archivos son rásters georreferenciados estándar EPSG:4326. Pueden arrastrarse directamente a la vista de QGIS o ArcGIS. Debido a las pirámides COG internas, el renderizado es instantáneo incluso con zoom interactivo.
