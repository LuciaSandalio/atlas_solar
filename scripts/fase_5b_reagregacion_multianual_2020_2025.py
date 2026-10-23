#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 5B: Reagregación Multianual Definitiva (Sexenio 2020–2025)
=============================================================
Convenio ATLAS II — FCEFyN (UNC) & OHMC — Gobierno de la Provincia de Córdoba

Combina los 6 años anuales canónicos (2020, 2021, 2022, 2023, 2024 y 2025):
1. Promedio climatológico sexenal ponderado uniforme (N = 48.645 celdas interiores).
2. Ciclo mensual sexenal (12 meses x 48.645 celdas).
3. Análisis de convergencia asintótica (Delta 2 a Delta 6) y CV interanual.
4. Matriz de correlación espacial interanual 6x6.
5. Exportación de productos consolidados a data/ready/atlas_definitivo_2km/.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
READY_DIR = DATA_DIR / "ready"
DEF_DIR = READY_DIR / "atlas_definitivo_2km"
METADATA_DIR = ROOT_DIR / "metadata"


def main():
    print("=" * 80)
    print("📈 REAGREGACIÓN MULTIANUAL CLIMATOLÓGICA (2020–2025)")
    print("=" * 80)

    years = [2020, 2021, 2022, 2023, 2024, 2025]
    annual_dfs = {}

    for y in years:
        fpath = DEF_DIR / f"atlas_anual_{y}.parquet"
        if not fpath.exists():
            raise FileNotFoundError(f"Falta archivo anual: {fpath}")
        df = pd.read_parquet(fpath)
        annual_dfs[y] = df
        print(f"   • Cargado año {y}: {len(df):,} celdas | GHI prov: {df.loc[df['in_province'], 'ghi_kwh_m2'].mean():.1f} kWh/m²/año")

    base_df = annual_dfs[2020][['cell_id', 'lat', 'lon', 'in_province']].copy()
    in_prov = base_df['in_province'].values

    # Inyectar regiones solares k=3 si existe
    reg_file = DEF_DIR / "atlas_anual_2020_2024_medio.parquet"
    if reg_file.exists():
        df_old = pd.read_parquet(reg_file)
        if 'region' in df_old.columns:
            base_df['region'] = df_old['region']
        else:
            base_df['region'] = 2
    else:
        base_df['region'] = 2

    # Promedios de variables solares y fotovoltaicas
    vars_to_average = [
        ('ghi_kwh_m2', 'ghi_anual_kwh_m2'),
        ('dni_kwh_m2', 'dni_anual_kwh_m2'),
        ('dhi_kwh_m2', 'dhi_anual_kwh_m2'),
        ('poa_fixed_kwh_m2', 'poa_fijo_anual_kwh_m2'),
        ('poa_tracker_kwh_m2', 'poa_tracker_anual_kwh_m2'),
        ('yf_fixed_kwh_kwp', 'yf_fixed_kwh_kwp'),
        ('yf_tracker_kwh_kwp', 'yf_tracker_kwh_kwp')
    ]

    for col_in, col_out in vars_to_average:
        arr_stack = np.column_stack([annual_dfs[y][col_in].values for y in years])
        base_df[col_out] = np.mean(arr_stack, axis=1)

    # Métricas derivadas
    base_df['cf_fixed_pct'] = 100.0 * base_df['yf_fixed_kwh_kwp'] / 8760.0
    base_df['cf_tracker_pct'] = 100.0 * base_df['yf_tracker_kwh_kwp'] / 8760.0
    base_df['ganancia_tracker_pct'] = 100.0 * (base_df['yf_tracker_kwh_kwp'] - base_df['yf_fixed_kwh_kwp']) / np.maximum(base_df['yf_fixed_kwh_kwp'], 1.0)

    out_medio = DEF_DIR / "atlas_anual_2020_2025_medio.parquet"
    base_df.to_parquet(out_medio, index=False)
    print(f"\n✓ Dataset sexenal oficial guardado en: {out_medio.name}")

    # Convergencia temporal
    ghi_stack = np.column_stack([annual_dfs[y]['ghi_kwh_m2'].values for y in years])
    m_1 = ghi_stack[:, 0]
    m_2 = np.mean(ghi_stack[:, :2], axis=1)
    m_3 = np.mean(ghi_stack[:, :3], axis=1)
    m_4 = np.mean(ghi_stack[:, :4], axis=1)
    m_5 = np.mean(ghi_stack[:, :5], axis=1)
    m_6 = np.mean(ghi_stack[:, :6], axis=1)

    delta_2 = 100.0 * (m_2 - m_1) / np.maximum(m_1, 1.0)
    delta_3 = 100.0 * (m_3 - m_2) / np.maximum(m_2, 1.0)
    delta_4 = 100.0 * (m_4 - m_3) / np.maximum(m_3, 1.0)
    delta_5 = 100.0 * (m_5 - m_4) / np.maximum(m_4, 1.0)
    delta_6 = 100.0 * (m_6 - m_5) / np.maximum(m_5, 1.0)

    cv_inter = 100.0 * (np.std(ghi_stack, axis=1) / np.maximum(m_6, 1.0))

    df_conv = pd.DataFrame({
        'cell_id': base_df['cell_id'],
        'lat': base_df['lat'],
        'lon': base_df['lon'],
        'in_province': base_df['in_province'],
        'M_2020': m_1,
        'M_2020_2021': m_2,
        'M_2020_2022': m_3,
        'M_2020_2023': m_4,
        'M_2020_2024': m_5,
        'M_2020_2025': m_6,
        'delta_2_pct': delta_2,
        'delta_3_pct': delta_3,
        'delta_4_pct': delta_4,
        'delta_5_pct': delta_5,
        'delta_6_pct': delta_6,
        'cv_interanual_pct': cv_inter
    })
    df_conv.to_parquet(DEF_DIR / "atlas_convergencia_temporal_6years.parquet", index=False)
    print("✓ Dataset de convergencia sexenal guardado en: atlas_convergencia_temporal_6years.parquet")

    # Ciclo mensual sexenal
    monthly_arrays = []
    for y in years:
        npy_path = DEF_DIR / f"atlas_monthly_{y}.npy"
        if npy_path.exists():
            arr_m = np.load(npy_path)
            monthly_arrays.append(arr_m)

    if len(monthly_arrays) == len(years):
        stacked_m = np.stack(monthly_arrays, axis=0)  # (6, N, 12)
        mean_m = np.mean(stacked_m, axis=0)          # (N, 12)
        month_names = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        df_months = pd.DataFrame({
            'cell_id': base_df['cell_id'],
            'lat': base_df['lat'],
            'lon': base_df['lon'],
            'in_province': base_df['in_province'],
            **{month_names[i]: mean_m[:, i] for i in range(12)}
        })
        df_months.to_parquet(DEF_DIR / "atlas_promedios_mensuales_sexenal.parquet", index=False)
        print("✓ Ciclo mensual sexenal guardado en: atlas_promedios_mensuales_sexenal.parquet")

    # Matriz de correlación espacial 6x6
    df_corr = pd.DataFrame({
        f'GHI_{y}': annual_dfs[y].loc[in_prov, 'ghi_kwh_m2'].values for y in years
    })
    corr_mat = df_corr.corr().round(4)
    corr_json = {
        'years': years,
        'columns': list(corr_mat.columns),
        'matrix': corr_mat.values.tolist(),
        'min_r': float(corr_mat.values[np.triu_indices(6, k=1)].min()),
        'max_r': float(corr_mat.values[np.triu_indices(6, k=1)].max()),
        'mean_r': float(corr_mat.values[np.triu_indices(6, k=1)].mean())
    }
    with open(METADATA_DIR / "matriz_correlacion_espacial_6x6.json", "w", encoding="utf-8") as f:
        json.dump(corr_json, f, indent=2)
    print("✓ Matriz de correlación espacial 6x6 guardada en: metadata/matriz_correlacion_espacial_6x6.json")


if __name__ == '__main__':
    main()

