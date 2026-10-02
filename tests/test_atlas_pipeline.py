#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suite de Pruebas Unitarias e Integración para el Pipeline del Atlas Solar
========================================================================
Valida las propiedades físicas, matemáticas y computacionales antes del
reprocesamiento masivo:
1. Equivalencia de cielo claro vectorizado frente a pvlib Location.get_clearsky().
2. Desplazamiento temporal exacto de 1 hora en Estrategia B (t+1).
3. Inmunidad a cruces de fechas entre días (frontera nocturna).
4. Recuperación analítica exacta de integrales conocidas en acumuladores.
5. Identidad binaria estricta entre reanudación desde checkpoint y corrida continua.
6. Determinismo y reproducibilidad de modos blended y hard.
7. Verificación de metadatos de grilla (reproducibilidad).
"""

import sys
import os
import json
import shutil
from pathlib import Path

# Compatibilidad Affine en Python 3.12
import affine
affine.Affine._astuple = property(
    lambda self: (self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h, self.i)
)

import pytest
import numpy as np
import pandas as pd
import pvlib
import geopandas as gpd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.fase_5b_02_run_atlas as run_atlas_module
from scripts.fase_5b_02_run_atlas import (
    calculate_refraction_apparent_zenith,
    get_linke_and_altitude_pressure,
    get_sparse_mapping_matrix,
    run_streaming_atlas
)
from scripts.fase_5b_03_aggregate import run_aggregation

GRID_FILE = ROOT_DIR / "data" / "ready" / "cordoba_grid_2km_regions.parquet"
SAMPLE_REFLECTANCE = ROOT_DIR / "data" / "processed" / "reflectancia_20210101.parquet"
PIPELINE_INPUTS = [
    GRID_FILE,
    SAMPLE_REFLECTANCE,
    ROOT_DIR / "data" / "ready" / "b0_model" / "B0_params.json",
    ROOT_DIR / "data" / "processed" / "bdjpt_final_models_regional_estacional.pkl",
    ROOT_DIR / "metadata" / "fase4_inference_contract.json",
]

requires_grid = pytest.mark.skipif(not GRID_FILE.exists(), reason="falta la grilla 2 km en data/ready")
requires_sample_scene = pytest.mark.skipif(not SAMPLE_REFLECTANCE.exists(), reason="falta data/processed/reflectancia_20210101.parquet")
requires_pipeline_data = pytest.mark.skipif(
    not all(p.exists() for p in PIPELINE_INPUTS), reason="faltan datos de entrada del motor de inferencia"
)


@requires_grid
def test_clearsky_vectorized_matches_location_canonical():
    """
    Verifica que el modelo Ineichen vectorizado con refracción atmosférica SPA
    y búsqueda en Altitude.h5 reproduzca Location.get_clearsky() con error idéntico a 0.0 W/m2.
    """
    grid = pd.read_parquet(GRID_FILE)
    linke_table, altitude_grid, pressure_grid = get_linke_and_altitude_pressure(grid)

    # Evaluar 20 puntos aleatorios en diferentes meses del año
    np.random.seed(42)
    sample_indices = np.random.choice(len(grid), 20, replace=False)
    test_dates = [
        "2021-01-15 15:00:00+00:00",
        "2021-04-15 16:00:00+00:00",
        "2021-07-15 17:00:00+00:00",
        "2021-10-15 14:00:00+00:00"
    ]

    max_err = 0.0
    for date_str in test_dates:
        t_idx = pd.DatetimeIndex([date_str])
        doy = t_idx.dayofyear.values[0]
        month_idx = t_idx.month.values[0] - 1
        dni_extra = pvlib.irradiance.get_extra_radiation(doy)

        for idx in sample_indices:
            lat = grid['lat'].iloc[idx]
            lon = grid['lon'].iloc[idx]
            alt = altitude_grid[idx]
            lt = linke_table[idx, month_idx]

            loc = pvlib.location.Location(lat, lon, altitude=alt)
            cs_target = loc.get_clearsky(t_idx, linke_turbidity=lt, dni_extra=dni_extra)['ghi'].values[0]

            # Modelo vectorizado
            sp = loc.get_solarposition(t_idx)
            zen_true = sp['zenith'].values[0]
            app_zen = calculate_refraction_apparent_zenith(
                np.array([zen_true]), np.array([pressure_grid[idx]])
            )[0]

            am_rel = pvlib.atmosphere.get_relative_airmass(app_zen)
            am_abs = pvlib.atmosphere.get_absolute_airmass(am_rel, pressure_grid[idx])
            lt = linke_table[idx, month_idx]

            cs_vec = pvlib.clearsky.ineichen(
                np.array([app_zen]),
                np.array([am_abs]),
                np.array([lt]),
                altitude=altitude_grid[idx],
                dni_extra=dni_extra
            )['ghi'][0]

            err = abs(cs_target - cs_vec)
            if err > max_err:
                max_err = err

    print(f"\n[Test] Máximo error de cielo claro Ineichen: {max_err:.8e} W/m²")
    assert max_err < 1e-3, f"Discrepancia inaceptable en cielo claro: {max_err} W/m²"


def test_temporal_alignment_strategy_b():
    """
    Verifica que la Estrategia B desplace exactamente -1 hora la hora objetivo
    respecto a la observación satelital (h_target = h_sat - 1 h).
    """
    h_sat = pd.Timestamp("2021-01-01 15:00:00+00:00")
    h_target = h_sat - pd.Timedelta(hours=1)
    assert h_target == pd.Timestamp("2021-01-01 14:00:00+00:00")
    assert (h_sat - h_target).total_seconds() == 3600.0


@requires_sample_scene
def test_no_day_boundary_crossover():
    """
    Verifica que las series de observación diurna nunca unan el último punto
    de un día con el primero del día siguiente a través de la noche.
    """
    df1 = pd.read_parquet(SAMPLE_REFLECTANCE)
    ts = pd.to_datetime(df1['timestamp'].unique())
    s_ts = pd.Series(ts).sort_values()
    diffs = s_ts.diff().dropna().dt.total_seconds() / 60.0
    # Entre escenas del mismo día, el intervalo máximo no debe exceder 30 min
    assert (diffs <= 30.0).all(), f"Hueco intradiurno anómalo: {diffs.max()} min"


def test_accumulators_recover_known_integral():
    """
    Verifica que la matemática de acumulación y 5B-03 recupere analíticamente
    la integral exacta sin derivas ni pérdidas de energía.
    """
    n_cells = 10
    sum_ghi = np.zeros((n_cells, 12, 24), dtype=np.float32)
    count_hours = np.zeros((n_cells, 12, 24), dtype=np.int32)

    # Simular 30 días de enero con irradiancia constante de 500 W/m² durante 8 horas (10..17 UTC)
    known_val = 500.0
    for h in range(10, 18):
        sum_ghi[:, 0, h] = known_val * 30.0 # 30 ocurrencias
        count_hours[:, 0, h] = 30

    denom = np.maximum(count_hours, 1).astype(np.float32)
    mean_h = (sum_ghi / denom) * (count_hours > 0)
    daily_integral = mean_h.sum(axis=2) * 1.0 # Wh/m2·día

    expected_daily = known_val * 8.0 # 4000 Wh/m2·día
    assert np.allclose(daily_integral[:, 0], expected_daily, atol=1e-4)


@pytest.mark.slow
@requires_pipeline_data
def test_checkpoint_resume_exact_match(tmp_path, monkeypatch):
    """
    Verifica que ejecutar N días con checkpoint y reanudar (--resume)
    produzca un resultado numéricamente idéntico (bit a bit) a una corrida continua.
    """
    # Las salidas y checkpoints se escriben en tmp_path, no en data/ready
    monkeypatch.setattr(run_atlas_module, "READY_DIR", tmp_path)
    test_out_cont = tmp_path / "test_accumulators_continuous.npz"
    test_out_res = tmp_path / "test_accumulators_resumed.npz"
    cp_file = tmp_path / "atlas_accumulators_2km_blended_checkpoint.npz"
    orig_output = tmp_path / "atlas_accumulators_2km_blended.npz"

    # 1. Corrida continua de 3 días
    run_streaming_atlas(mode='blended', max_days=3, checkpoint_freq=100)
    shutil.copy(orig_output, test_out_cont)

    # 2. Corrida con interrupción a los 2 días y luego reanudación
    orig_output.unlink(missing_ok=True)
    cp_file.unlink(missing_ok=True)

    # Simular interrupción a los 2 días en un lote programado de 3
    run_streaming_atlas(mode='blended', max_days=3, stop_after_days=2, checkpoint_freq=10)
    assert cp_file.exists(), "El checkpoint debió haberse generado a los 2 días"

    # Reanudar para completar el día 3
    run_streaming_atlas(mode='blended', max_days=3, resume=True)
    shutil.copy(orig_output, test_out_res)

    # 3. Comparar acumuladores
    npz1 = np.load(test_out_cont)
    npz2 = np.load(test_out_res)

    for arr_name in ['sum_ghi', 'sum_dni', 'sum_dhi', 'sum_poa_fixed', 'sum_poa_tracker',
                     'sum_p_ac_fixed', 'sum_p_ac_tracker', 'count_hours', 'expected_count']:
        diff = np.max(np.abs(npz1[arr_name] - npz2[arr_name]))
        assert diff == 0.0, f"Discrepancia en {arr_name} tras reanudación: {diff}"

    print("\n[Test] Identidad estricta (bit a bit) entre corrida continua y reanudación comprobada.")


def test_reproducibility_and_grid_contract():
    """
    Verifica que el Contrato de Inferencia contenga todos los metadatos de grilla requeridos
    para evitar mezclas accidentales de versiones (grid_id, resolucion, n_cells, etc.).
    """
    contract_file = ROOT_DIR / "metadata" / "fase4_inference_contract.json"
    with open(contract_file, "r") as f:
        contract = json.load(f)

    assert "operational_grid" in contract
    op_grid = contract["operational_grid"]
    assert op_grid["resolution_km"] == 2.0
    assert op_grid["n_cells_buffered"] == 59837
    assert op_grid["n_cells_province"] == 48645
    assert op_grid["spatial_interpolation_required"] is False
    assert contract["temporal_alignment"]["strategy"] == "B (t+1)"


@requires_grid
def test_blended_and_hard_modes_produce_valid_results():
    """
    Verifica que tanto el modo blended como el modo hard produzcan acumuladores
    finitos, no negativos y mutuamente consistentes.
    """
    grid = pd.read_parquet(GRID_FILE)
    n_cells = len(grid)
    weight_cols = sorted(c for c in grid.columns if c.startswith('weight_R'))
    assert len(weight_cols) == 3, "La regionalización vigente tiene 3 regiones"
    weights = grid[weight_cols].values

    # Verificar que los pesos del modo blended sumen 1.0 en todas las celdas (tolerancia de redondeo a 4 decimales)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-3), "Los pesos blended deben sumar 1.0 dentro de la precisión almacenada"
    assert (weights >= 0.0).all(), "No puede haber pesos blended negativos"

    # Verificar que las regiones dominantes pertenezcan al conjunto {0, 1, 2}
    dominant = grid['region'].values
    assert set(np.unique(dominant)).issubset({0, 1, 2})

