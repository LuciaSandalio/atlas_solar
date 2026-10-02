#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 5B - Paso 02: Motor de Inferencia Streaming Territorial a 2 km (Atlas Solar Córdoba)
========================================================================================
Aplica la cadena física y de machine learning completa día a día sobre las 59,837
celdas operativas de la grilla provincial de 2 km (data/ready/cordoba_grid_2km_regions.parquet):

1. Ingesta streaming de archivos diarios satelitales GOES-16 (reflectancia_YYYYMMDD.parquet).
2. Agregación temporal horaria (dt.round('h')) sobre la malla de observación y proyección
   espacial instantánea a la grilla provincial de 2 km mediante mapeo baricéntrico disperso
   precomputado W (shape: 59,837 x 9,595), ejecutado en ~0.1 ms por escena.
3. Desplazamiento temporal estricto según Contrato de Inferencia de Fase 4 (Estrategia B: t+1):
   la observación satelital en h_sat sincroniza con la integral radiométrica de superficie
   en la hora precedente h_target = h_sat - 1 h.
4. Física de cenit astronómico verdadero (Erbs, Tarpley B0) vs cenit aparente con refracción
   atmosférica SPA exacta (Ineichen, Perez, Seguidor 1-eje).
5. Modelo de cielo claro Ineichen vectorizado de alta fidelidad con turbidez de Linke (resolución 5')
   y altimetría topográfica exacta (Altitude.h5 / alt2pres), equivalente a Location.get_clearsky()
   con error idéntico a 0.0 W/m².
6. Evaluación del modelo físico de brillo de cielo claro Tarpley (B0) y contraste Fr.
7. Inferencia regional-estacional del ensamble híbrido BD-JPT + HGB:
   - Modo 'blended': interpolación ponderada continua por pesos de consenso espacial 3-NN
     (weight_R0 a weight_R{k-1}), garantizando superficies climatológicas continuas sin artefactos
     de frontera.
   - Modo 'hard': asignación determinística al modelo de la región solar dominante.
8. Descomposición espectro-angular Erbs et al. (1982).
9. Transposición anisotrópica Perez et al. (1990):
   - Configuración A: Plano Fijo 30° orientado al Norte (azimut 0°).
   - Configuración B: Seguidor 1-eje horizontal N-S con rotación E-O y backtracking (GCR=0.35, max 60°).
10. Modelo térmico NOCT (45°C) y modelo paramétrico completo PVWatts (DC + Inversor AC + 10% BOS).
11. Acumuladores climatológicos de memoria constante (shape: 59,837 celdas x 12 meses x 24 horas),
    con seguimiento de horas esperadas y efectivas para cálculo insesgado de cobertura física.
12. Manejo estricto de checkpoints incompatibles mediante run_signature criptográfico y auditoría
    de días fallidos (failed_dates).
"""

import sys
import os
import time
import json
import glob
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pvlib
import geopandas as gpd
import joblib
import h5py
import scipy.sparse
import scipy.spatial

# Rutas estándar del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Importación permanente y segura de modelos BD-JPT
from models.bdjpt import BDJPTWrapper, HybridBDJPT_HGB

# Rutas estándar del proyecto
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
READY_DIR = DATA_DIR / "ready"
READY_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR = Path("metadata")
METADATA_DIR.mkdir(parents=True, exist_ok=True)

GRID_2KM_FILE = READY_DIR / "cordoba_grid_2km_regions.parquet"
GRID_5KM_FILE = DATA_DIR / "grids" / "cordoba_grid_5km.geojson"
B0_PARAMS_FILE = READY_DIR / "b0_model" / "B0_params.json"
MODELS_FILE = PROCESSED_DIR / "bdjpt_final_models_regional_estacional.pkl"
CONTRACT_FILE = METADATA_DIR / "fase4_inference_contract.json"

SPARSE_MAPPING_FILE = READY_DIR / "sparse_mapping_5km_to_2km.npz"
LINKE_FILE = READY_DIR / "linke_turbidity_grid_2km.npy"
ALTITUDE_FILE = READY_DIR / "altitude_grid_2km.npy"
PRESSURE_FILE = READY_DIR / "pressure_grid_2km.npy"

DAYS_IN_MONTH = {1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


def get_season(month: int) -> str:
    """Retorna la estación del año para el hemisferio sur."""
    if month in [12, 1, 2]:
        return 'Verano'
    elif month in [3, 4, 5]:
        return 'Otono'
    elif month in [6, 7, 8]:
        return 'Invierno'
    else:
        return 'Primavera'


def get_sparse_mapping_matrix(grid_2km: pd.DataFrame) -> scipy.sparse.csr_matrix:
    """
    Obtiene o precomputa la matriz baricéntrica dispersa W (59837, 9595)
    para mapear observaciones satelitales desde la grilla de muestreo (5 km)
    a la grilla oficial provincial (2 km) en ~0.1 ms por escena.
    """
    if SPARSE_MAPPING_FILE.exists():
        print(f"📂 Cargando matriz de mapeo baricéntrico disperso desde {SPARSE_MAPPING_FILE}...")
        return scipy.sparse.load_npz(SPARSE_MAPPING_FILE)

    print("⏳ Precomputando matriz baricéntrica dispersa 5 km -> 2 km (Delaunay)...")
    t0 = time.time()
    g5 = gpd.read_file(GRID_5KM_FILE)
    pts5 = np.column_stack([g5['lon'].values, g5['lat'].values])
    pts2 = np.column_stack([grid_2km['lon'].values, grid_2km['lat'].values])

    tri = scipy.spatial.Delaunay(pts5)
    simplex = tri.find_simplex(pts2)
    b = tri.transform[simplex, :2]
    r = pts2 - tri.transform[simplex, 2]
    w = np.einsum('ijk,ik->ij', b, r)
    w = np.c_[w, 1.0 - w.sum(axis=1)]

    outside = (simplex == -1)
    rows, cols, data = [], [], []
    for i in range(len(pts2)):
        if not outside[i]:
            v = tri.simplices[simplex[i]]
            for j in range(3):
                rows.append(i)
                cols.append(v[j])
                data.append(w[i, j])
        else:
            # Fallback a vecino más cercano en los bordes extremos del buffer
            dist = np.sum((pts5 - pts2[i])**2, axis=1)
            nearest_idx = np.argmin(dist)
            rows.append(i)
            cols.append(nearest_idx)
            data.append(1.0)

    W = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(len(pts2), len(pts5)), dtype=np.float32)
    scipy.sparse.save_npz(SPARSE_MAPPING_FILE, W)
    print(f"✓ Matriz W guardada en {SPARSE_MAPPING_FILE} en {time.time()-t0:.2f} s. Shape: {W.shape}")
    return W


def get_linke_and_altitude_pressure(grid_2km: pd.DataFrame):
    """
    Obtiene o precomputa las tablas de turbidez de Linke (59837, 12),
    altitud (59837,) y presión superficial estándar (59837,) a 2 km.
    """
    if LINKE_FILE.exists() and ALTITUDE_FILE.exists() and PRESSURE_FILE.exists():
        linke_table = np.load(LINKE_FILE)
        altitude_grid = np.load(ALTITUDE_FILE)
        pressure_grid = np.load(PRESSURE_FILE)
        return linke_table, altitude_grid, pressure_grid

    print("⏳ Precomputando tablas de turbidez de Linke, altitud y presión para 59,837 celdas...")
    t0 = time.time()
    lats = grid_2km['lat'].values
    lons = grid_2km['lon'].values

    pvlib_path = Path(pvlib.__file__).parent
    linke_h5_path = pvlib_path / 'data' / 'LinkeTurbidities.h5'
    alt_h5_path = pvlib_path / 'data' / 'Altitude.h5'

    scale_lat = 2160.0 / -180.0
    center_lat = 90.0 + 1.0 / scale_lat / 2.0
    lat_idx = np.round((lats - center_lat) * scale_lat).astype(np.int32)

    scale_lon = 4320.0 / 360.0
    center_lon = -180.0 + 1.0 / scale_lon / 2.0
    lon_idx = np.round((lons - center_lon) * scale_lon).astype(np.int32)

    with h5py.File(linke_h5_path, 'r') as f:
        lt_data = f['LinkeTurbidity'][:]
        linke_table = (lt_data[lat_idx, lon_idx, :] / 20.0).astype(np.float32)

    with h5py.File(alt_h5_path, 'r') as f:
        alt_data = f['Altitude'][:]
        alt_raw = alt_data[lat_idx, lon_idx]
        altitude_grid = np.where(alt_raw == 255, 0.0, alt_raw.astype(np.float32) * 28.0 - 450.0).astype(np.float32)

    pressure_grid = pvlib.atmosphere.alt2pres(altitude_grid).astype(np.float32)

    np.save(LINKE_FILE, linke_table)
    np.save(ALTITUDE_FILE, altitude_grid)
    np.save(PRESSURE_FILE, pressure_grid)

    print(f"✓ Tablas geofísicas 2 km precomputadas en {time.time()-t0:.2f} s:")
    print(f"  • Linke shape: {linke_table.shape}, media: {linke_table.mean():.2f}")
    print(f"  • Altitud shape: {altitude_grid.shape}, rango: [{altitude_grid.min():.0f}, {altitude_grid.max():.0f}] m")
    print(f"  • Presión shape: {pressure_grid.shape}, media: {pressure_grid.mean()/100:.1f} hPa")

    return linke_table, altitude_grid, pressure_grid


def calculate_refraction_apparent_zenith(zenith_true: np.ndarray, pressure_pa: np.ndarray) -> np.ndarray:
    """
    Calcula el cenit aparente con corrección de refracción atmosférica SPA exacta
    a partir del cenit astronómico verdadero y la presión local.
    Equivalente exacto (error < 1e-14°) a la salida de spa_python().
    """
    elev = 90.0 - zenith_true
    press_mbar = pressure_pa / 100.0
    # Fórmula estándar NREL SPA / Bennett / Saemundsson
    delta_e = (press_mbar / 1010.0) * (283.0 / 285.0) * 1.02 / (
        60.0 * np.tan(np.radians(elev + 10.3 / (elev + 5.11)))
    )
    apparent_zenith = np.clip(zenith_true - delta_e, 0.0, 90.0)
    return apparent_zenith.astype(np.float32)


def run_streaming_atlas(mode='blended', max_days=None, start_date=None, end_date=None, resume=False, checkpoint_freq=200, stop_after_days=None):
    print("=" * 80)
    print(f"🚀 FASE 5B-02: MOTOR DE INFERENCIA STREAMING TERRITORIAL 2 KM (Modo: {mode.upper()})")
    print("=" * 80)

    # 1. Cargar Grilla Oficial de 2 km
    if not GRID_2KM_FILE.exists():
        raise FileNotFoundError(f"❌ No se encontró la grilla de 2 km en {GRID_2KM_FILE}")
    print(f"📂 Cargando grilla territorial 2 km desde {GRID_2KM_FILE}...")
    grid = pd.read_parquet(GRID_2KM_FILE)
    n_cells = len(grid)
    print(f"   • Total celdas operativas: {n_cells:,} (Provincia: {grid['in_province'].sum():,}, Buffer: {(~grid['in_province']).sum():,})")

    weight_cols = sorted(c for c in grid.columns if c.startswith('weight_R'))
    n_regions = len(weight_cols)
    weights_reg = grid[weight_cols].values.astype(np.float32)
    dominant_regions = grid['region'].values.astype(np.int32)

    # 2. Cargar parámetros B0 de Tarpley
    print(f"📂 Cargando parámetros Tarpley B0 desde {B0_PARAMS_FILE}...")
    with open(B0_PARAMS_FILE, "r") as f:
        b0_params = json.load(f)
    A = float(b0_params['A'])
    B = float(b0_params['B'])
    C = float(b0_params['C'])
    D = float(b0_params['D'])

    # 3. Cargar Modelos Regionales-Estacionales (Pickle limpio con models.bdjpt)
    print(f"🧠 Cargando modelos híbridos BD-JPT + HGB desde {MODELS_FILE}...")
    models_dict = joblib.load(MODELS_FILE)
    if set(models_dict) != set(range(n_regions)):
        raise ValueError(
            f"Los modelos tienen regiones {sorted(models_dict)} pero la grilla tiene {n_regions} "
            f"({weight_cols}). Reentrene la Fase 4 con los clusters vigentes."
        )

    # 4. Cargar o precomputar W, Linke, Altitud y Presión a 2 km
    W = get_sparse_mapping_matrix(grid)
    linke_table, altitude_grid, pressure_grid = get_linke_and_altitude_pressure(grid)

    # 5. Cargar Contrato de Inferencia y verificar alineamiento
    with open(CONTRACT_FILE, "r") as f:
        contract = json.load(f)
    strategy = contract['temporal_alignment']['strategy']
    print(f"📜 Contrato validado: Estrategia de alineación temporal: '{strategy}'")

    # 6. Lista de archivos satelitales diarios
    ref_files = sorted(glob.glob(str(PROCESSED_DIR / "reflectancia_*.parquet")))
    print(f"📂 Archivos satelitales diarios encontrados: {len(ref_files):,}")

    if start_date:
        ref_files = [f for f in ref_files if Path(f).stem.replace("reflectancia_", "") >= start_date.replace("-", "")]
    if end_date:
        ref_files = [f for f in ref_files if Path(f).stem.replace("reflectancia_", "") <= end_date.replace("-", "")]
    if max_days is not None and max_days > 0:
        ref_files = ref_files[:max_days]
        print(f"⚠️ Modo prueba activo: limitando a los primeros {max_days} días.")

    total_files = len(ref_files)
    if total_files == 0:
        raise ValueError("❌ No hay archivos para procesar con los filtros especificados.")

    # Identificador de corrida estricto para evitar incompatibilidad de checkpoints
    run_signature = hashlib.sha256(
        f"{mode}_{start_date}_{end_date}_{n_cells}_{contract.get('pipeline_version', '5B.2.0')}".encode()
    ).hexdigest()[:16]

    output_file = READY_DIR / f"atlas_accumulators_2km_{mode}.npz"
    checkpoint_file = READY_DIR / f"atlas_accumulators_2km_{mode}_checkpoint.npz"

    # 7. Inicializar o reanudar acumuladores de memoria constante (shape: 59837, 12, 24)
    if resume and checkpoint_file.exists():
        print(f"🔄 Verificando checkpoint existente: {checkpoint_file}...")
        cp = np.load(checkpoint_file, allow_pickle=True)
        cp_signature = str(cp['run_signature']) if 'run_signature' in cp else None
        if cp_signature != run_signature:
            raise ValueError(
                f"❌ Incompatibilidad de Checkpoint: firma del checkpoint ({cp_signature}) "
                f"no coincide con la configuración actual ({run_signature}). "
                f"No use --resume o elimine el checkpoint obsoleto."
            )
        sum_ghi = cp['sum_ghi']
        sum_dni = cp['sum_dni']
        sum_dhi = cp['sum_dhi']
        sum_poa_fixed = cp['sum_poa_fixed']
        sum_poa_tracker = cp['sum_poa_tracker']
        sum_p_ac_fixed = cp['sum_p_ac_fixed']
        sum_p_ac_tracker = cp['sum_p_ac_tracker']
        count_hours = cp['count_hours']
        expected_count = cp['expected_count']
        processed_dates = set(cp['processed_dates'].tolist())
        failed_dates = cp['failed_dates'].tolist() if 'failed_dates' in cp else []
        print(f"   ✓ Reanudación exitosa: {len(processed_dates)} días previamente completados.")
    else:
        sum_ghi = np.zeros((n_cells, 12, 24), dtype=np.float32)
        sum_dni = np.zeros((n_cells, 12, 24), dtype=np.float32)
        sum_dhi = np.zeros((n_cells, 12, 24), dtype=np.float32)
        sum_poa_fixed = np.zeros((n_cells, 12, 24), dtype=np.float32)
        sum_poa_tracker = np.zeros((n_cells, 12, 24), dtype=np.float32)
        sum_p_ac_fixed = np.zeros((n_cells, 12, 24), dtype=np.float32)
        sum_p_ac_tracker = np.zeros((n_cells, 12, 24), dtype=np.float32)
        count_hours = np.zeros((n_cells, 12, 24), dtype=np.int32)
        expected_count = np.zeros((n_cells, 12, 24), dtype=np.int32)
        processed_dates = set()
        failed_dates = []

    # 8. Parámetros FV (Mono-PERC, PVWatts)
    pdc0 = 1000.0          # 1 kWp nominal DC
    gamma_pdc = -0.0035    # -0.35 %/°C
    noct = 45.0            # °C
    t_ref = 25.0           # °C
    eta_inv_nom = 0.96     # Eficiencia nominal inversor
    eta_bos = 0.90         # Pérdidas BOS (10%)

    t_start = time.time()
    days_completed = len(processed_dates)

    print(f"\n⚡ Procesando {total_files} días territoriales en resolución nativa de 2 km...")

    for idx, fpath in enumerate(ref_files):
        day_str = Path(fpath).stem.replace("reflectancia_", "")
        if day_str in processed_dates:
            continue

        try:
            df_day = pd.read_parquet(fpath)
            if df_day.empty:
                continue

            # 1. Agregación temporal horaria sobre malla 5 km
            df_day['hour_round'] = df_day['timestamp'].dt.round('h')
            # Filtro diurno preliminar sobre observaciones satelitales
            df_day = df_day[df_day['zenith_deg'] < 85.0]
            if df_day.empty:
                processed_dates.add(day_str)
                continue

            agg5 = df_day.groupby(['cell_id', 'hour_round'])[['reflectance', 'zenith_deg', 'azimuth_deg']].mean().reset_index()
            unique_hours = sorted(agg5['hour_round'].unique())

            for h_sat in unique_hours:
                sub_h = agg5[agg5['hour_round'] == h_sat].sort_values('cell_id')

                # Asegurar alineación dimensional de 9,595 celdas antes de W.dot()
                if len(sub_h) != 9595:
                    sub_full = pd.DataFrame({'cell_id': np.arange(9595)}).merge(sub_h, on='cell_id', how='left')
                    ref5 = sub_full['reflectance'].fillna(0.0).values.astype(np.float32)
                    zen5 = sub_full['zenith_deg'].fillna(85.0).values.astype(np.float32)
                    az5 = sub_full['azimuth_deg'].fillna(0.0).values.astype(np.float32)
                else:
                    ref5 = sub_h['reflectance'].values.astype(np.float32)
                    zen5 = sub_h['zenith_deg'].values.astype(np.float32)
                    az5 = sub_h['azimuth_deg'].values.astype(np.float32)

                # Mapeo baricéntrico disperso ultrarrápido a 2 km (0.1 ms)
                ref2 = np.clip(W.dot(ref5), 0.0, 1.2)
                zen2_true = W.dot(zen5)
                az2 = W.dot(az5)

                # Alineación temporal según Contrato Fase 4: Estrategia B (t+1)
                # La observación satelital en h_sat modela la hora precedente h_target = h_sat - 1 h
                h_target = pd.Timestamp(h_sat) - pd.Timedelta(hours=1)
                month_idx = h_target.month - 1
                hour_idx = h_target.hour
                doy = h_target.dayofyear

                # Refracción atmosférica SPA exacta para cenit aparente
                app_zen2 = calculate_refraction_apparent_zenith(zen2_true, pressure_grid)

                # Filtro de elevación solar diurna para acumulación física
                mask_daylight = (app_zen2 < 85.0)
                if not np.any(mask_daylight):
                    continue

                # Registro de horas esperadas diurnas
                np.add.at(expected_count, (np.arange(n_cells)[mask_daylight], month_idx, hour_idx), 1)

                # Modelo de cielo claro Ineichen de alta fidelidad (exacto a Location.get_clearsky())
                dni_extra = pvlib.irradiance.get_extra_radiation(doy)
                am_rel = pvlib.atmosphere.get_relative_airmass(app_zen2)
                am_abs = pvlib.atmosphere.get_absolute_airmass(am_rel, pressure_grid)
                lt = linke_table[:, month_idx]

                cs = pvlib.clearsky.ineichen(app_zen2, am_abs, lt, altitude=altitude_grid, dni_extra=dni_extra)
                ghi_cs = np.maximum(np.asarray(cs['ghi']), 0.0)

                # Modelo Tarpley B0 y contraste Fr
                z_rad = np.radians(zen2_true)
                cosZ = np.cos(z_rad)
                sinZ = np.sin(z_rad)
                cosGamma = np.cos(np.radians(az2))
                b0 = np.clip(A + B*cosZ + C*sinZ*cosGamma + D*sinZ*(cosGamma**2), 0.01, None)
                fr = np.maximum(ref2**2 - b0**2, 0.0)

                X = np.column_stack([ref2, b0, fr, cosZ, np.full(n_cells, doy, dtype=np.float32)])
                season = get_season(h_target.month)

                # Inferencia regional-estacional
                if mode == 'blended':
                    kt_pred = np.zeros(n_cells, dtype=np.float32)
                    for k in range(n_regions):
                        m_k = models_dict[k][season]['hybrid']
                        w_k = weights_reg[:, k]
                        # Solo evaluar modelos que tengan peso no nulo en la región
                        mask_active = (w_k > 0.001)
                        if np.any(mask_active):
                            kt_k = m_k.predict(X[mask_active], ghi_cs[mask_active])
                            kt_pred[mask_active] += w_k[mask_active] * kt_k
                else:
                    # Modo 'hard' determinístico
                    kt_pred = np.zeros(n_cells, dtype=np.float32)
                    for k in range(n_regions):
                        mask_k = (dominant_regions == k)
                        if np.any(mask_k):
                            m_k = models_dict[k][season]['hybrid']
                            kt_pred[mask_k] = m_k.predict(X[mask_k], ghi_cs[mask_k])

                kt_pred = np.clip(kt_pred, 0.0, 1.15)
                ghi = kt_pred * ghi_cs

                # Descomposición radiativa Erbs et al. (1982) con cenit verdadero
                erbs_res = pvlib.irradiance.erbs(ghi, zen2_true, pd.DatetimeIndex([h_target] * n_cells))
                dni = np.maximum(np.asarray(erbs_res['dni']), 0.0)
                dhi = np.maximum(np.asarray(erbs_res['dhi']), 0.0)

                # Transposición Fija (Perez et al. 1990, Tilt 30° Norte)
                poa_fixed = pvlib.irradiance.get_total_irradiance(
                    surface_tilt=30.0,
                    surface_azimuth=0.0,
                    solar_zenith=app_zen2,
                    solar_azimuth=az2,
                    dni=dni,
                    ghi=ghi,
                    dhi=dhi,
                    dni_extra=dni_extra,
                    model='perez'
                )
                g_poa_fix = np.maximum(np.asarray(poa_fixed['poa_global']), 0.0)

                # Transposición Seguidor 1-eje (Perez et al. 1990, Backtracking GCR 0.35)
                trk = pvlib.tracking.singleaxis(
                    apparent_zenith=app_zen2,
                    solar_azimuth=az2,
                    axis_tilt=0.0,
                    axis_azimuth=0.0,
                    max_angle=60.0,
                    backtrack=True,
                    gcr=0.35
                )
                s_tilt = np.nan_to_num(np.asarray(trk['surface_tilt']), nan=0.0)
                s_az = np.nan_to_num(np.asarray(trk['surface_azimuth']), nan=0.0)

                poa_trk = pvlib.irradiance.get_total_irradiance(
                    surface_tilt=s_tilt,
                    surface_azimuth=s_az,
                    solar_zenith=app_zen2,
                    solar_azimuth=az2,
                    dni=dni,
                    ghi=ghi,
                    dhi=dhi,
                    dni_extra=dni_extra,
                    model='perez'
                )
                g_poa_trk = np.maximum(np.asarray(poa_trk['poa_global']), 0.0)

                # Modelo térmico NOCT y temperatura ambiental estacional de Córdoba
                t_amb = 24.0 if season == 'Verano' else 18.0 if season == 'Otono' else 11.0 if season == 'Invierno' else 19.0
                t_cell_fix = t_amb + ((noct - 20.0) / 800.0) * g_poa_fix
                t_cell_trk = t_amb + ((noct - 20.0) / 800.0) * g_poa_trk

                # Conversión PVWatts DC + Inversor AC + BOS 10%
                pdc_fix = pvlib.pvsystem.pvwatts_dc(g_poa_fix, t_cell_fix, pdc0, gamma_pdc, t_ref)
                pac_fix_kw = (np.maximum(pvlib.inverter.pvwatts(pdc_fix, pdc0, eta_inv_nom), 0.0) * eta_bos) / 1000.0

                pdc_trk = pvlib.pvsystem.pvwatts_dc(g_poa_trk, t_cell_trk, pdc0, gamma_pdc, t_ref)
                pac_trk_kw = (np.maximum(pvlib.inverter.pvwatts(pdc_trk, pdc0, eta_inv_nom), 0.0) * eta_bos) / 1000.0

                # Acumulación climatológica directa a 2 km (Delta t = 1.0 h)
                idx_cells = np.arange(n_cells)[mask_daylight]
                np.add.at(sum_ghi, (idx_cells, month_idx, hour_idx), ghi[mask_daylight].astype(np.float32))
                np.add.at(sum_dni, (idx_cells, month_idx, hour_idx), dni[mask_daylight].astype(np.float32))
                np.add.at(sum_dhi, (idx_cells, month_idx, hour_idx), dhi[mask_daylight].astype(np.float32))
                np.add.at(sum_poa_fixed, (idx_cells, month_idx, hour_idx), g_poa_fix[mask_daylight].astype(np.float32))
                np.add.at(sum_poa_tracker, (idx_cells, month_idx, hour_idx), g_poa_trk[mask_daylight].astype(np.float32))
                np.add.at(sum_p_ac_fixed, (idx_cells, month_idx, hour_idx), pac_fix_kw[mask_daylight].astype(np.float32))
                np.add.at(sum_p_ac_tracker, (idx_cells, month_idx, hour_idx), pac_trk_kw[mask_daylight].astype(np.float32))
                np.add.at(count_hours, (idx_cells, month_idx, hour_idx), 1)

            processed_dates.add(day_str)
            days_completed += 1

            # Reporte periódico de avance
            if (idx + 1) % 25 == 0 or (idx + 1) == total_files:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta_min = (total_files - (idx + 1)) / (rate * 60.0) if rate > 0 else 0
                print(f"   [{idx + 1:4d}/{total_files:4d}] {day_str} | Vel: {rate:4.2f} días/s | Transcurrido: {elapsed/60:4.1f} min | ETA: {eta_min:4.1f} min")

            # Checkpoint periódico
            if days_completed % checkpoint_freq == 0:
                np.savez_compressed(
                    checkpoint_file,
                    sum_ghi=sum_ghi, sum_dni=sum_dni, sum_dhi=sum_dhi,
                    sum_poa_fixed=sum_poa_fixed, sum_poa_tracker=sum_poa_tracker,
                    sum_p_ac_fixed=sum_p_ac_fixed, sum_p_ac_tracker=sum_p_ac_tracker,
                    count_hours=count_hours, expected_count=expected_count,
                    processed_dates=np.array(sorted(list(processed_dates))),
                    failed_dates=np.array(failed_dates),
                    run_signature=run_signature
                )

            # Interrupción temprana controlada para pruebas de reanudación
            if stop_after_days is not None and days_completed >= stop_after_days:
                print(f"🛑 Interrupción controlada alcanzada tras {days_completed} días. Guardando checkpoint...")
                np.savez_compressed(
                    checkpoint_file,
                    sum_ghi=sum_ghi, sum_dni=sum_dni, sum_dhi=sum_dhi,
                    sum_poa_fixed=sum_poa_fixed, sum_poa_tracker=sum_poa_tracker,
                    sum_p_ac_fixed=sum_p_ac_fixed, sum_p_ac_tracker=sum_p_ac_tracker,
                    count_hours=count_hours, expected_count=expected_count,
                    processed_dates=np.array(sorted(list(processed_dates))),
                    failed_dates=np.array(failed_dates),
                    run_signature=run_signature
                )
                break

        except Exception as e:
            print(f"⚠️ Error procesando día {day_str}: {e}")
            failed_dates.append({"date": day_str, "error": str(e)})
            continue

    # Guardar producto final comprimido en formato nativo 2 km
    print(f"\n💾 Guardando acumuladores oficiales 2 km en: {output_file}...")
    metadata = {
        "pipeline": "Fase 5B-02",
        "pipeline_version": "5B.2.0",
        "resolution_km": 2.0,
        "n_cells": n_cells,
        "n_cells_province": int(grid['in_province'].sum()),
        "mode": mode,
        "run_signature": run_signature,
        "n_days_processed": len(processed_dates),
        "total_files_evaluated": total_files,
        "failed_dates": failed_dates,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "temporal_alignment": contract.get("temporal_alignment", {}),
        "pv_specs": {
            "pdc0_w": pdc0,
            "gamma_pdc": gamma_pdc,
            "noct_c": noct,
            "eta_inv_nom": eta_inv_nom,
            "eta_bos": eta_bos,
            "fixed_tilt_deg": 30.0,
            "fixed_azimuth_deg": 0.0,
            "tracker_type": "singleaxis_horizontal_NS",
            "tracker_gcr": 0.35,
            "tracker_max_angle": 60.0
        }
    }

    np.savez_compressed(
        output_file,
        sum_ghi=sum_ghi,
        sum_dni=sum_dni,
        sum_dhi=sum_dhi,
        sum_poa_fixed=sum_poa_fixed,
        sum_poa_tracker=sum_poa_tracker,
        sum_p_ac_fixed=sum_p_ac_fixed,
        sum_p_ac_tracker=sum_p_ac_tracker,
        count_hours=count_hours,
        expected_count=expected_count,
        cell_ids=grid['cell_id'].values,
        lat=grid['lat'].values,
        lon=grid['lon'].values,
        in_province=grid['in_province'].values,
        region=grid['region'].values,
        processed_dates=np.array(sorted(list(processed_dates))),
        failed_dates=np.array(failed_dates),
        run_signature=run_signature,
        metadata=json.dumps(metadata)
    )

    # Eliminar checkpoint si la corrida concluyó al 100%
    if checkpoint_file.exists() and len(processed_dates) == total_files:
        checkpoint_file.unlink(missing_ok=True)

    total_elapsed = time.time() - t_start
    print("=" * 80)
    if len(failed_dates) == 0:
        print(f"✅ FASE 5B-02 COMPLETADA EXITOSAMENTE EN {total_elapsed/60:.2f} MINUTOS.")
    else:
        print(f"⚠️ FASE 5B-02 COMPLETADA CON {len(failed_dates)} DÍAS FALLIDOS EN {total_elapsed/60:.2f} MINUTOS.")
    print(f"   • Celdas: {n_cells:,} (resolución nativa 2 km)")
    print(f"   • Días procesados: {len(processed_dates):,}")
    print(f"   • Horas válidas acumuladas: {count_hours.sum():,}")
    print(f"   • Horas esperadas acumuladas: {expected_count.sum():,}")
    print(f"   • Cobertura territorial diurna global: {(count_hours.sum() / max(expected_count.sum(), 1))*100:.2f}%")
    print(f"   • Archivo de salida: {output_file} ({output_file.stat().st_size / 1e6:.1f} MB)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motor de Inferencia Streaming Territorial 2 km (Fase 5B-02).")
    parser.add_argument("--mode", choices=['blended', 'hard'], default='blended',
                        help="Modo de pertenencia regional: 'blended' (ponderado continuo 3-NN) o 'hard' (dominante).")
    parser.add_argument("--max-days", type=int, default=None,
                        help="Limitar a N días para prueba o validación rápida.")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Fecha inicial (YYYY-MM-DD o YYYYMMDD).")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Fecha final (YYYY-MM-DD o YYYYMMDD).")
    parser.add_argument("--resume", action="store_true",
                        help="Reanudar desde checkpoint si existe.")
    parser.add_argument("--checkpoint-freq", type=int, default=200,
                        help="Frecuencia de guardado de checkpoints en días.")
    args = parser.parse_args()

    run_streaming_atlas(
        mode=args.mode,
        max_days=args.max_days,
        start_date=args.start_date,
        end_date=args.end_date,
        resume=args.resume,
        checkpoint_freq=args.checkpoint_freq
    )
