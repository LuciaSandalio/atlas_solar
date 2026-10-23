# ATLAS Solar de la Provincia de Córdoba

Convenio ATLAS II (FCEFyN/UNC y OHMC). Estima irradiancia solar (GHI, DNI, DHI) y potencial fotovoltaico sobre una grilla de 2 km a partir de imágenes GOES-16 y una red de estaciones de superficie.

## Estado de validación

**Producción definitiva bloqueada.** Los productos existentes no tienen trazabilidad
suficiente para certificar el pipeline regional-estacional 3 × 4. El contrato marca
`BLOCKED_UNVERIFIED_PIPELINE`; no se reemplazaron los resultados históricos.
La inferencia, agregación, auditoría aprobatoria, exportación y generación del visor
antiguas ahora fallan explícitamente antes de producir nuevas salidas.

```bash
python scripts/audit_atlas_release.py --output metadata/atlas_release_audit.json
```

Este comando devuelve código 1 mientras existan bloqueos. No aprueba el Atlas.
Consultar [la auditoría y sus criterios](docs/auditoria_pipeline_atlas.md).

Existe un entrenamiento **candidato**, separado de producción, con estrategia C
compartida, doce híbridos, validación espacial por estación y holdout temporal:

```bash
python scripts/train_atlas_candidate.py --output data/ready/atlas_candidate_nuevo
```

La carpeta de salida debe ser nueva. Guarda modelos, predicciones, métricas y un
contrato con huellas de las entradas. No modifica el contrato operativo ni elimina
las limitaciones de procedencia de las features históricas. Los resultados de la
corrida local están en `data/ready/atlas_candidate_20261005_v2/`.

El extractor masivo corregido lee la proyección de cada NetCDF GOES-16/19 y registra
satélite, archivo fuente, huellas y geometría. Su nueva salida predeterminada es
`data/processed_2km_v2/`, con archivos `.provenance.json` que deben conservarse junto
a cada Parquet. Rechaza sobrescribir extracciones antiguas y devuelve error si falla
un día. El período predeterminado ahora incluye diciembre de 2025. La convención
radiométrica anterior se conserva explícitamente como pendiente de validación;
tener metadatos no certifica cobertura anual ni calibración entre satélites.

## Entorno

```bash
conda env create -f solar.yml
conda activate atlas_solar
```

Todos los scripts se ejecutan desde la raíz del repositorio (`python scripts/<script>.py`). Los datos pesados (`data/processed*`, `data/ready`, `data/geometry`, `data/goes`) no están versionados.

## Pipeline

La tabla siguiente describe las etapas históricas; no implica que estén habilitadas
o validadas para generar el producto definitivo.

| Fase | Scripts | Descripción |
|---|---|---|
| 1 | `generate_grid_2km.py`, `fase_1_b_grid2km.py`, `fase_1_b_mass_grid2km.py` | Grilla de 2 km y extracción de radiancia/reflectancia GOES-16 |
| 2 | `fase_2_*.py`, `get_atm_data.py` | QC de estaciones Davis/Pilar, cielo claro REST2, calibración virtual y clustering solar |
| 3 | `fase_3_b0.py` | Brillo de cielo despejado (B0) |
| 4 | `fase_4_bdjpt.py` | Modelo BD-JPT + HGB regional-estacional con validación cruzada espacial |
| 5A | `fase_5_pv_modeling.py` | Conversión GHI a potencial fotovoltaico |
| 5B | `fase_5b_00_audit.py` a `fase_5b_05_export_geotiff_preliminar.py` | Inferencia territorial a 2 km, agregación y exportación GeoTIFF/GeoPackage |
| Validación | `validate_5a_5b_consistency.py` | Consistencia 5A y 5B |

`scripts/dev_step*.py` son pasos de desarrollo/auditoría y `scripts/archive/` contiene scripts históricos. Los parámetros del modelo congelado están en `metadata/fase4_inference_contract.json`.

## Visor web

```bash
python viewer/serve.py             # consulta los productos históricos existentes
```

## Tests

```bash
pytest tests
```

Los tests requieren los datos locales de `data/ready` y `data/processed`.
