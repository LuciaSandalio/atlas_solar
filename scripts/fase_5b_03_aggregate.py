#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 5B - Paso 03: Agregación Climatológica Mensual y Anual a 2 km
=================================================================
Toma los acumuladores de memoria constante de Fase 5B-02 (shape: 59,837 x 12 x 24)
e integra de manera robusta e insesgada las climatologías provinciales a 2 km:

1. Ciclo diurno medio horario por celda y mes.
2. Integrales diarias medias mensuales (PSH en kWh/m2·día y kWh/kWp·día).
3. Totales mensuales ponderados por días reales de mes (año medio 365.25 d).
4. Totales anuales climatológicos:
   - GHI, DNI, DHI (kWh/m2·año).
   - POA Fijo 30° y Seguidor 1-eje (kWh/m2·año).
   - Rendimiento Específico Yf Fijo y Seguidor (kWh/kWp·año).
   - Ganancia de seguimiento radiativa y energética (%).
   - Factores de Capacidad CF Fijo y Seguidor (%).
   - Cobertura física real (valid_hours / expected_hours) sin ceros silenciosos.
5. Exportación de productos analíticos en resolución nativa de 2 km:
   - data/ready/atlas_climatology_annual_2km_{mode}.parquet
   - data/ready/atlas_climatology_monthly_2km_{mode}.parquet
6. Resumen estadístico provincial y regional ponderado por área (coseno de latitud)
   restringido estrictamente al territorio provincial (excluyendo celdas buffer).
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

DATA_DIR = Path("data")
READY_DIR = DATA_DIR / "ready"
REPORTES_DIR = Path("reportes/figures")
REPORTES_DIR.mkdir(parents=True, exist_ok=True)

GRID_2KM_FILE = READY_DIR / "cordoba_grid_2km_regions.parquet"
DAYS_IN_MONTH = {1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
MONTH_NAMES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
REGION_NAMES = {0: "Región 0 (Sierras Altas)", 1: "Región 1 (Sierras Bajas / Transición)", 2: "Región 2 (Llanura Pampeana)"}


def run_aggregation(mode='blended'):
    print("=" * 80)
    print(f"📊 FASE 5B-03: AGREGACIÓN CLIMATOLÓGICA TERRITORIAL 2 KM (Modo: {mode.upper()})")
    print("=" * 80)

    input_file = READY_DIR / f"atlas_accumulators_2km_{mode}.npz"
    if not input_file.exists():
        raise FileNotFoundError(f"❌ No se encontró el archivo de acumuladores 2 km: {input_file}")

    print(f"📂 Cargando acumuladores desde {input_file}...")
    npz = np.load(input_file, allow_pickle=True)

    sum_ghi = npz['sum_ghi']
    sum_dni = npz['sum_dni']
    sum_dhi = npz['sum_dhi']
    sum_poa_fixed = npz['sum_poa_fixed']
    sum_poa_tracker = npz['sum_poa_tracker']
    sum_p_ac_fixed = npz['sum_p_ac_fixed']
    sum_p_ac_tracker = npz['sum_p_ac_tracker']
    count_hours = npz['count_hours']
    expected_count = npz['expected_count'] if 'expected_count' in npz else None

    cell_ids = npz['cell_ids']
    lats = npz['lat']
    lons = npz['lon']
    in_province = npz['in_province'] if 'in_province' in npz else np.ones(len(cell_ids), dtype=bool)
    regions = npz['region']
    processed_dates = npz['processed_dates']
    metadata = json.loads(str(npz['metadata'])) if 'metadata' in npz else {}

    n_cells = len(cell_ids)
    n_days = len(processed_dates)
    print(f"   • Celdas: {n_cells:,} (Provinciales: {in_province.sum():,}, Buffer: {(~in_province).sum():,})")
    print(f"   • Días registrados: {n_days:,} ({processed_dates[0]} a {processed_dates[-1]})")
    print(f"   • Horas válidas acumuladas: {count_hours.sum():,}")

    # 1. Ciclo diurno horario medio: (n_cells, 12, 24)
    # Evitar ceros silenciosos en horas con observaciones
    denom = np.maximum(count_hours, 1).astype(np.float32)

    mean_h_ghi = sum_ghi / denom
    mean_h_dni = sum_dni / denom
    mean_h_dhi = sum_dhi / denom
    mean_h_poa_fix = sum_poa_fixed / denom
    mean_h_poa_trk = sum_poa_tracker / denom
    mean_h_pac_fix = sum_p_ac_fixed / denom
    mean_h_pac_trk = sum_p_ac_tracker / denom

    # En horas nocturnas o sin datos, anular
    valid_mask = (count_hours > 0)
    mean_h_ghi *= valid_mask
    mean_h_dni *= valid_mask
    mean_h_dhi *= valid_mask
    mean_h_poa_fix *= valid_mask
    mean_h_poa_trk *= valid_mask
    mean_h_pac_fix *= valid_mask
    mean_h_pac_trk *= valid_mask

    # 2. Integral diaria promedio mensual: suma sobre las 24 horas (Delta t = 1.0 h)
    # Forma: (n_cells, 12)
    daily_m_ghi = mean_h_ghi.sum(axis=2) * 1.0         # Wh/m2·día
    daily_m_dni = mean_h_dni.sum(axis=2) * 1.0
    daily_m_dhi = mean_h_dhi.sum(axis=2) * 1.0
    daily_m_poa_fix = mean_h_poa_fix.sum(axis=2) * 1.0
    daily_m_poa_trk = mean_h_poa_trk.sum(axis=2) * 1.0
    daily_m_pac_fix = mean_h_pac_fix.sum(axis=2) * 1.0 # kWh/kWp·día
    daily_m_pac_trk = mean_h_pac_trk.sum(axis=2) * 1.0

    # 3. Totales mensuales ponderados por días de mes
    days_arr = np.array([DAYS_IN_MONTH[m] for m in range(1, 13)], dtype=np.float32) # (12,)

    monthly_ghi_kwh = (daily_m_ghi * days_arr) / 1000.0         # kWh/m2·mes
    monthly_dni_kwh = (daily_m_dni * days_arr) / 1000.0
    monthly_dhi_kwh = (daily_m_dhi * days_arr) / 1000.0
    monthly_poa_fix_kwh = (daily_m_poa_fix * days_arr) / 1000.0
    monthly_poa_trk_kwh = (daily_m_poa_trk * days_arr) / 1000.0
    monthly_yf_fix = daily_m_pac_fix * days_arr                # kWh/kWp·mes
    monthly_yf_trk = daily_m_pac_trk * days_arr

    # 4. Totales anuales climatológicos (suma sobre los 12 meses)
    ann_ghi_kwh = monthly_ghi_kwh.sum(axis=1)                  # kWh/m2·año
    ann_dni_kwh = monthly_dni_kwh.sum(axis=1)
    ann_dhi_kwh = monthly_dhi_kwh.sum(axis=1)
    ann_poa_fix_kwh = monthly_poa_fix_kwh.sum(axis=1)
    ann_poa_trk_kwh = monthly_poa_trk_kwh.sum(axis=1)
    ann_yf_fix = monthly_yf_fix.sum(axis=1)                    # kWh/kWp·año
    ann_yf_trk = monthly_yf_trk.sum(axis=1)

    # Métricas derivadas
    psh_ghi = ann_ghi_kwh / 365.25                             # PSH kWh/m2·día
    psh_poa_fix = ann_poa_fix_kwh / 365.25
    psh_poa_trk = ann_poa_trk_kwh / 365.25

    cf_fix = (ann_yf_fix / 8760.0) * 100.0                     # %
    cf_trk = (ann_yf_trk / 8760.0) * 100.0

    gain_rad_trk = ((ann_poa_trk_kwh - ann_poa_fix_kwh) / np.maximum(ann_poa_fix_kwh, 1e-3)) * 100.0
    gain_yf_trk = ((ann_yf_trk - ann_yf_fix) / np.maximum(ann_yf_fix, 1e-3)) * 100.0

    # Cobertura física real
    valid_hours_per_cell = count_hours.sum(axis=(1, 2))
    if expected_count is not None:
        expected_hours_per_cell = expected_count.sum(axis=(1, 2))
        coverage_pct = np.where(expected_hours_per_cell > 0,
                                (valid_hours_per_cell / expected_hours_per_cell) * 100.0,
                                100.0)
    else:
        expected_hours_per_cell = np.full(n_cells, n_days * 12.0)
        coverage_pct = np.clip((valid_hours_per_cell / expected_hours_per_cell) * 100.0, 0.0, 100.0)

    # 5. Construcción del DataFrame Anual a 2 km
    df_annual = pd.DataFrame({
        "cell_id": cell_ids,
        "lat": lats,
        "lon": lons,
        "in_province": in_province,
        "region": regions,
        "region_name": [REGION_NAMES.get(r, f"Región {r}") for r in regions],
        "ghi_kwh_m2_yr": np.round(ann_ghi_kwh, 1),
        "dni_kwh_m2_yr": np.round(ann_dni_kwh, 1),
        "dhi_kwh_m2_yr": np.round(ann_dhi_kwh, 1),
        "psh_ghi_kwh_m2_d": np.round(psh_ghi, 2),
        "poa_fixed_kwh_m2_yr": np.round(ann_poa_fix_kwh, 1),
        "poa_tracker_kwh_m2_yr": np.round(ann_poa_trk_kwh, 1),
        "psh_fixed_kwh_m2_d": np.round(psh_poa_fix, 2),
        "psh_tracker_kwh_m2_d": np.round(psh_poa_trk, 2),
        "yf_fixed_kwh_kwp_yr": np.round(ann_yf_fix, 1),
        "yf_tracker_kwh_kwp_yr": np.round(ann_yf_trk, 1),
        "cf_fixed_pct": np.round(cf_fix, 2),
        "cf_tracker_pct": np.round(cf_trk, 2),
        "gain_rad_tracker_pct": np.round(gain_rad_trk, 2),
        "gain_energy_tracker_pct": np.round(gain_yf_trk, 2),
        "n_valid_hours": valid_hours_per_cell,
        "n_expected_hours": expected_hours_per_cell,
        "coverage_pct": np.round(coverage_pct, 1)
    })

    annual_parquet = READY_DIR / f"atlas_climatology_annual_2km_{mode}.parquet"
    df_annual.to_parquet(annual_parquet, index=False)
    print(f"\n💾 Climatología Anual 2 km guardada en: {annual_parquet}")

    # 6. Construcción del DataFrame Mensual a 2 km
    monthly_rows = []
    for c in range(n_cells):
        for m in range(12):
            monthly_rows.append({
                "cell_id": cell_ids[c],
                "in_province": in_province[c],
                "month": m + 1,
                "month_name": MONTH_NAMES[m],
                "ghi_kwh_m2_mo": round(float(monthly_ghi_kwh[c, m]), 2),
                "dni_kwh_m2_mo": round(float(monthly_dni_kwh[c, m]), 2),
                "dhi_kwh_m2_mo": round(float(monthly_dhi_kwh[c, m]), 2),
                "poa_fixed_kwh_m2_mo": round(float(monthly_poa_fix_kwh[c, m]), 2),
                "poa_tracker_kwh_m2_mo": round(float(monthly_poa_trk_kwh[c, m]), 2),
                "yf_fixed_kwh_kwp_mo": round(float(monthly_yf_fix[c, m]), 2),
                "yf_tracker_kwh_kwp_mo": round(float(monthly_yf_trk[c, m]), 2),
                "psh_ghi_d": round(float(daily_m_ghi[c, m] / 1000.0), 2),
                "psh_fixed_d": round(float(daily_m_poa_fix[c, m] / 1000.0), 2),
                "psh_tracker_d": round(float(daily_m_poa_trk[c, m] / 1000.0), 2)
            })
    df_monthly = pd.DataFrame(monthly_rows)
    monthly_parquet = READY_DIR / f"atlas_climatology_monthly_2km_{mode}.parquet"
    df_monthly.to_parquet(monthly_parquet, index=False)
    print(f"💾 Climatología Mensual 2 km guardada en: {monthly_parquet}")

    # 7. Resumen Provincial Estricto (Territorio Provincial, excluyendo Buffer)
    print("\n" + "=" * 80)
    print("📈 RESUMEN DEL RECURSO SOLAR Y POTENCIAL FV PROVINCIAL (TERRITORIO CÓRDOBA)")
    print("=" * 80)

    # Filtrar estrictamente a territorio provincial
    df_prov = df_annual[df_annual['in_province']].copy()
    weights_area = np.cos(np.radians(df_prov['lat'].values))

    def weighted_mean(series, w):
        return np.sum(series * w) / np.sum(w)

    def weighted_std(series, w):
        wm = weighted_mean(series, w)
        return np.sqrt(np.sum(w * (series - wm)**2) / np.sum(w))

    summary_list = []
    for r in sorted(df_prov['region'].unique()):
        sub = df_prov[df_prov['region'] == r]
        w_sub = np.cos(np.radians(sub['lat'].values))
        r_name = REGION_NAMES.get(r, f"Región {r}")
        summary_list.append({
            "Región": r_name,
            "Celdas": f"{len(sub):,}",
            "GHI (kWh/m$^2$·a)": f"{weighted_mean(sub['ghi_kwh_m2_yr'], w_sub):.1f} ± {weighted_std(sub['ghi_kwh_m2_yr'], w_sub):.1f}",
            "PSH (kWh/m$^2$·d)": f"{weighted_mean(sub['psh_ghi_kwh_m2_d'], w_sub):.2f}",
            "POA Fijo (kWh/m$^2$·a)": f"{weighted_mean(sub['poa_fixed_kwh_m2_yr'], w_sub):.1f}",
            "POA Tracker (kWh/m$^2$·a)": f"{weighted_mean(sub['poa_tracker_kwh_m2_yr'], w_sub):.1f}",
            "Yf Fijo (kWh/kWp·a)": f"{weighted_mean(sub['yf_fixed_kwh_kwp_yr'], w_sub):.1f}",
            "Yf Tracker (kWh/kWp·a)": f"{weighted_mean(sub['yf_tracker_kwh_kwp_yr'], w_sub):.1f}",
            "CF Fijo (%)": f"{weighted_mean(sub['cf_fixed_pct'], w_sub):.1f}\\%",
            "CF Tracker (%)": f"{weighted_mean(sub['cf_tracker_pct'], w_sub):.1f}\\%",
            "Ganancia Tracker (%)": f"+{weighted_mean(sub['gain_energy_tracker_pct'], w_sub):.2f}\\%"
        })

    # Fila provincial total ponderada por área
    summary_list.append({
        "Región": "\\textbf{Total Provincial}",
        "Celdas": f"\\textbf{{{len(df_prov):,}}}",
        "GHI (kWh/m$^2$·a)": f"\\textbf{{{weighted_mean(df_prov['ghi_kwh_m2_yr'], weights_area):.1f} ± {weighted_std(df_prov['ghi_kwh_m2_yr'], weights_area):.1f}}}",
        "PSH (kWh/m$^2$·d)": f"\\textbf{{{weighted_mean(df_prov['psh_ghi_kwh_m2_d'], weights_area):.2f}}}",
        "POA Fijo (kWh/m$^2$·a)": f"\\textbf{{{weighted_mean(df_prov['poa_fixed_kwh_m2_yr'], weights_area):.1f}}}",
        "POA Tracker (kWh/m$^2$·a)": f"\\textbf{{{weighted_mean(df_prov['poa_tracker_kwh_m2_yr'], weights_area):.1f}}}",
        "Yf Fijo (kWh/kWp·a)": f"\\textbf{{{weighted_mean(df_prov['yf_fixed_kwh_kwp_yr'], weights_area):.1f}}}",
        "Yf Tracker (kWh/kWp·a)": f"\\textbf{{{weighted_mean(df_prov['yf_tracker_kwh_kwp_yr'], weights_area):.1f}}}",
        "CF Fijo (%)": f"\\textbf{{{weighted_mean(df_prov['cf_fixed_pct'], weights_area):.1f}\\%}}",
        "CF Tracker (%)": f"\\textbf{{{weighted_mean(df_prov['cf_tracker_pct'], weights_area):.1f}\\%}}",
        "Ganancia Tracker (%)": f"\\textbf{{+{weighted_mean(df_prov['gain_energy_tracker_pct'], weights_area):.2f}\\%}}"
    })

    df_summary = pd.DataFrame(summary_list)
    print(df_summary.to_string(index=False))

    tex_file = REPORTES_DIR / "tabla_fase5b_metricas_provinciales_consolidadas.tex"
    latex_table = "\\begin{table}[htpb]\n\\centering\n\\small\n"
    latex_table += "\\caption{Resumen territorial y regional del recurso solar y potencial fotovoltaico en Córdoba (Atlas Solar 2 km, Convenio ATLAS II).}\n"
    latex_table += "\\label{tab:fase5b_metricas_provinciales}\n"
    latex_table += "\\begin{tabular}{lcccccccc}\n\\toprule\n"
    latex_table += "Región & Celdas & GHI & PSH & $Y_f$ Fijo & $Y_f$ Tracker & CF Fijo & CF Tracker & Ganancia \\\\\n"
    latex_table += " & & (kWh/m$^2$·a) & (kWh/m$^2$·d) & (kWh/kWp·a) & (kWh/kWp·a) & (\\%) & (\\%) & (\\%) \\\\\n\\midrule\n"

    for row in summary_list:
        latex_table += f"{row['Región']} & {row['Celdas']} & {row['GHI (kWh/m$^2$·a)']} & {row['PSH (kWh/m$^2$·d)']} & {row['Yf Fijo (kWh/kWp·a)']} & {row['Yf Tracker (kWh/kWp·a)']} & {row['CF Fijo (%)']} & {row['CF Tracker (%)']} & {row['Ganancia Tracker (%)']} \\\\\n"

    latex_table += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"\n📄 Tabla LaTeX consolidada exportada a: {tex_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agregación Climatológica 2 km (Fase 5B-03).")
    parser.add_argument("--mode", choices=['blended', 'hard'], default='blended',
                        help="Modo de agregación: 'blended' o 'hard'.")
    args = parser.parse_args()
    run_aggregation(mode=args.mode)
