# Trazabilidad del producto final del Atlas

Fecha de revisión: 2026-10-05. Estado: **NO VERIFICADO para el pipeline solicitado**.

Esta revisión no certifica los productos existentes ni cambia sus valores.
Las etapas históricas incompatibles ahora tienen un bloqueo ejecutable. Este
documento define la evidencia necesaria para aceptar un producto y conserva la
instantánea inicial de la auditoría; las correcciones posteriores se detallan abajo.

## Correcciones y ejecución realizadas

- `models/atlas_contract.py`: definición única de C (redondeo horario, promedio
  ref/B0/cosZ de extremos consecutivos del mismo día UTC, Fr calculado después),
  validación de doce modelos ajustados y blending que conserva NaN.
- `scripts/train_atlas_candidate.py`: entrenamiento reproducible con 2020,
  GroupKFold por estación, holdout 2021 y Pilar excluida de todos los ajustes de
  híbridos. Usa las regiones actuales del GPKG: el mapa estación/celda histórico
  todavía contenía etiquetas de cuatro regiones y sólo se usa para sus cell_id,
  verificando sus coordenadas contra la grilla canónica.
- Candidato completo ejecutado en `data/ready/atlas_candidate_20261005_v2/`, con
  modelos, contrato, métricas por región/estación y predicciones OOF/holdout.
  Las huellas de entradas se comprueban antes y después del entrenamiento.
- Resultado agregado: R² GHI espacial 0,821485 (170.254 pares), temporal 2021
  0,757846 (198.609 pares), Pilar 0,924762 (5.122 pares). Estas métricas **nuevas**
  no justifican retrospectivamente los valores del contrato anterior.
- Limitaciones: B0 y regionalización previos no se reajustan dentro de los folds;
  no hay primavera en el holdout temporal; R0/invierno tiene R² temporal 0,189112;
  falta procedencia de escenas originales de las features históricas. Son métricas
  de un candidato condicionado por esas entradas, no una aprobación científica.
- Extractor: proyección leída de cada NetCDF, identidad del satélite/archivo por
  fila, huellas de grilla/código/escenas y sidecar enlazado al Parquet por SHA-256.
  Directorio predeterminado nuevo `data/processed_2km_v2`, sin sobrescribir los
  datos anteriores; incluye hasta 2025-12-31 y devuelve error ante días fallidos.
  La convención radiométrica histórica sigue pendiente de validación.
- Prueba con dos NetCDF reales descargados de NOAA, uno GOES-16 y otro GOES-19,
  sobre tres celdas canónicas cada uno: lectura, proyección, interpolación e
  identidad correctas. Evidencia en `metadata/atlas_goes_extraction_smoke.json`.
  Esto no valida calibración radiométrica entre satélites ni cobertura completa.
- El contrato operativo perdió la aprobación: estado `BLOCKED_UNVERIFIED_PIPELINE`.
  Las métricas y aprobación previas se preservan como afirmaciones históricas no
  verificadas. El modelo operativo no se reemplazó por el candidato.
- Las entradas históricas de entrenamiento, auditoría aprobatoria, inferencia,
  agregación, exportación y generación de visor fallan explícitamente. Los dos
  scripts definitivos antes vacíos también fallan, en vez de terminar con éxito.
  No existe todavía un reemplazo validado de inferencia/anuales/exportación.
- `scripts/audit_atlas_release.py` produce un diagnóstico JSON y sale con código 1.
  Es un preflight conservador que **nunca emite certificados de aprobación**;
  no basta editar un campo `status` para completar la validación científica.

Resultados estructurados: `metadata/atlas_candidate_validation.json` y
`metadata/atlas_release_audit.json`. Los datos diarios pueden seguir creciendo
durante otras extracciones: el recuento del JSON corresponde a su `audited_at`.

## Pipeline requerido

NetCDF ABI C02 → muestreo canónico y geometría solar UTC → B0 y cielo claro
radiativo → estrategia C del entrenamiento → 12 modelos regionales-estacionales
→ blending territorial → GHI → Erbs → DNI/DHI → Perez → POA → NOCT/PVWatts
→ productos anuales por celda → promedio multianual y estadísticos → exportaciones.

El modelo global existente no sustituye los 3 × 4 modelos solicitados. Las tres
regiones de una grilla o de un mapa no demuestran entrenamiento regional.

## Evidencia de la instantánea inicial, antes de las correcciones

| Elemento | Resultado |
|---|---|
| PKL regional-estacional histórico | Diccionario con regiones 0, 1, 2 y 3; cuatro estaciones por región. No corresponde a las tres regiones actuales. |
| PKL indicado por el contrato | Un único objeto `models.bdjpt.HybridBDJPT_HGB`. No contiene doce modelos. |
| Motor `fase_5b_02_run_atlas.py` | Carga el PKL histórico y B0 histórico mediante constantes. Conserva alineación B, entrada `data/processed` e interpolación desde 9.595 celdas. Leer el texto C del contrato no modifica estas operaciones. |
| Scripts definitivos | `fase_5b_atlas_definitivo_2020_2024.py` y `fase_5b_05_export_geotiff_definitivo.py` están vacíos. |
| Reconstrucción del entrenamiento de desarrollo | `dev_step1` a `dev_step4` están vacíos también en el índice de Git. No se identificó una cadena completa reproducible para el modelo global. |
| Extracción masiva actual | La proyección se fija en el inicializador; no satisface la lectura de la geometría propia de cada NetCDF solicitada. |
| Datos diarios locales | 366/365/365/365/366 archivos para 2020–2024; 113 para 2025. La presencia de archivos no demuestra cobertura intradiaria completa. |
| Parquet anuales definitivos | Existen 2020–2024. Cada uno tiene 59.837 identificadores únicos que coinciden con la grilla; las siete variables energéticas revisadas son finitas y no negativas. |
| Trazabilidad de esos Parquet | Los metadatos Arrow encontrados sólo contienen `pandas`; las columnas no incluyen cobertura, conteos de intervalos ni identificación de corrida/modelo. No se encontró un manifiesto de ejecución final entre los JSON de `data/ready`. |
| Media 2020–2024 | Para siete variables energéticas, difiere del promedio aritmético de los cinco Parquet anuales como máximo 0,050049 en sus unidades. Compatible con redondeo a una décima; no prueba el algoritmo generador. |
| Factor de capacidad | La tabla media difiere del promedio de CF anuales con 8.784 h en años bisiestos: hasta 0,027386 puntos porcentuales fijo y 0,032122 tracker. Debe reconstruirse desde los productos anuales según la convención solicitada. |
| GeoTIFF de GHI quinquenal | Coincide exactamente con la tabla media en los centroides de las 48.645 celdas provinciales. |
| Media de GHI del manifiesto del visor | Coincide con la media provincial de la tabla: aproximadamente 1.888,0349 kWh/m²/año. No se verificaron todas las capas ni `cell_database.js`. |
| Procedimiento de generación del visor | `prepare_viewer.py` todavía configura productos preliminares 2020/2021/2023; no documenta la generación del manifiesto quinquenal actual. |

La igualdad entre dos productos no identifica el modelo que los originó. Los
GeoTIFF revisados contienen etiquetas de período, proyecto y unidades, pero no
huellas del modelo, contrato o ejecución. El manifiesto de grilla
`metadata/run_manifest_2km.json` no es un manifiesto de ejecución del Atlas.

## Métricas y alcance de las comprobaciones

No se encontró respaldo para los R² 0,9166 y 0,8842 del contrato en los CSV
revisados. El generador histórico de esos campos utiliza una regresión lineal
ajustada y evaluada sobre los mismos pares, distinta de una validación independiente
del híbrido. No debe atribuirse esa definición a los valores actuales sin recuperar
su procedencia.

Los CSV existentes informan para opción C: R² global test 2021 = 0,046515 con
193.194 observaciones; Pilar = 0,349544 con 1.824 observaciones; R² GHI de validación
espacial 2020 = 0,703002 con 155.161 observaciones. Son resultados de experimentos
guardados, no una certificación del PKL actual: falta su vinculación verificable.

Las cinco pruebas rápidas aprobadas anteriormente no prueban el pipeline completo.
La prueba de escenas se omite por una ruta histórica y la integración pesada no se
ejecutó. No se reentrenó ni reprocesó el Atlas durante esta auditoría.

La lectura de centroides con `rasterio.sample` falló por incompatibilidad local de
`Affine`. La comparación GHI se completó resolviendo la transformación afín con
NumPy y leyendo los píxeles correspondientes, con comprobación de límites. No se
modificaron paquetes del entorno.

## Condiciones de aceptación del producto final

1. **Contrato generado por entrenamiento.** Debe identificar por SHA-256 los doce
   modelos, el B0, la grilla y los datos de entrenamiento; registrar código y
   dependencias, variables y unidades, particiones, métricas y número de pares.
   Ningún estado de aprobación escrito a mano sustituye esa evidencia.
2. **Reconstrucción satelital identificable.** Cada lote debe conservar satélite,
   escenas fuente, timestamps UTC, proyección y parámetros radiométricos leídos
   del NetCDF, identificadores canónicos y versión del extractor. Los Parquet
   previos sin esa procedencia requieren verificación o regeneración.
3. **Una transformación temporal compartida.** Entrenamiento e inferencia deben
   usar la misma función para C, indicando qué variables se promedian, etiquetas
   de intervalos y tratamiento de extremos ausentes. No cruzar noches ni reemplazar
   pares faltantes por observaciones únicas sin haber entrenado esa convención.
4. **Inferencia regional-estacional verificable.** Exactamente tres regiones por
   cuatro estaciones, pesos 3-NN normalizados y vinculados a las estaciones fuente.
   Deben probarse selección de estación y modos hard/blended con predicciones reales.
5. **Física y unidades comprobadas.** Distinguir GHI/GHI_cs del índice usado por
   Erbs; verificar cierre GHI = DNI cosZ + DHI, controles de sol bajo, parámetros
   Perez y tracker, temperatura ambiente documentada y conversión W → kWh para
   PVWatts. Registrar los parámetros especificados por el usuario en el contrato.
6. **Producto anual canónico.** Un Parquet por año y celda, con variables energéticas,
   CF calculado con las horas del año, pesos/región, cobertura válida y esperada,
   faltantes y estado QA. Noche física puede valer cero; ausencia de datos no.
   Un año incompleto no se acepta como año completo ni se anualiza silenciosamente.
7. **Agregación trazable.** Para 2020–2025, aceptar primero los seis años completos;
   promediar los productos anuales, incluidos CF y ganancia. Definir explícitamente
   el tratamiento de denominadores nulos y la convención de desviación del CV.
   El ciclo mensual debe provenir de productos mensuales identificados por año.
8. **Una fuente para todas las salidas.** Una tabla maestra derivada de esos Parquet
   alimenta GeoTIFF/COG, GPKG, figuras, estadísticas y ambas bases del visor. Cada
   salida registra su tabla y ejecución fuente. Verificar valores por `cell_id`,
   máscaras, unidades, nodata y redondeo, no solamente una media provincial.
9. **Validación de entrega.** Ejecutar comparación con estaciones independientes,
   hard/blended, límites regionales, extremos y Salinas Grandes; reproducir las
   tablas del informe. Registrar tolerancias justificadas y resultados de QA.
10. **Reproducción y aprobación.** Con código versionado y manifiestos de entradas,
    ejecutar un lote de prueba completo y luego los años aceptables en una carpeta
    de salida nueva. La exportación definitiva debe rechazar cobertura incompleta,
    modelos incompatibles o linaje ausente. Implementar y probar ese bloqueo antes
    de llamar definitivo al producto. Las rutas antiguas ya se bloquean, pero ese
    bloqueo no sustituye el nuevo motor ni su validación científica completa.

## Identificación de la instantánea auditada

Huellas SHA-256 para reconocer los archivos revisados; no certifican cómo se generaron:

| Archivo | SHA-256 |
|---|---|
| `metadata/fase4_inference_contract.json` | `758786d0149229ea1c65f6899965582e835ff52c0dfdd27a0fb2c219d105a8b3` |
| `data/ready/b0_model/model_hybrid_bdjpt_hgb_optionC.pkl` | `afc69ded8dec1e05890da916d0163af6e32a60aeefe43e83f0932fef9acd4220` |
| `data/processed/bdjpt_final_models_regional_estacional.pkl` | `526b24174836f1331f0f4c8ed17e7eaa8e3e4db3e8bbcaf7fa9e6f93ceb32ab8` |
| `data/ready/atlas_definitivo_2km/atlas_anual_2020_2024_medio.parquet` | `64490414a108d0f2aeb23082f541b3629d8d91e0ea60bc8435b087909781a6cb` |
| `data/ready/maps_geotiff/cordoba_ghi_anual_quinquenal_2km_cog.tif` | `673d69b6dfc077444e651cad429c94ed404758ee22592b20628ce1c6b3690151` |
| `viewer/data/manifest.json` | `cfcac623414c759f12087aaf1ed51fe9c9fab14b5152498110933898b9b4bcaf` |
