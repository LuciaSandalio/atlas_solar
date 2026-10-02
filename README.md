# ATLAS Solar de la Provincia de Córdoba

Convenio ATLAS II (FCEFyN/UNC y OHMC). Estima irradiancia solar (GHI, DNI, DHI) y potencial fotovoltaico sobre una grilla de 2 km a partir de imágenes GOES-16 y una red de estaciones de superficie.

## Entorno

```bash
conda env create -f solar.yml
conda activate atlas_solar
```

Todos los scripts se ejecutan desde la raíz del repositorio (`python scripts/<script>.py`). Los datos pesados (`data/processed*`, `data/ready`, `data/geometry`, `data/goes`) no están versionados.

## Pipeline

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
python scripts/prepare_viewer.py   # genera viewer/data y viewer/overlays desde los GeoTIFF
python viewer/serve.py             # http://localhost:8080/viewer/index.html
```

## Tests

```bash
pytest tests
```

Los tests requieren los datos locales de `data/ready` y `data/processed`.
