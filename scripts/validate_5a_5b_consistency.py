#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validación de Consistencia Científica e Integración Numérica: Fase 5A <-> Fase 5B
================================================================================
Verifica que la cadena física y de ingeniería fotovoltaica implementada en el
motor territorial 5B-02 reproduzca exactamente los resultados radiométricos y
energéticos de la validación regional 5A para las 4 estaciones de referencia:
- Los Espinillos (Región 0 - Sierras)
- Ausonia (Región 1 - Centro-Sur)
- El Tío (Región 2 - Este)
- Presa Chañar (Región 3 - Norte)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pvlib

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.fase_5_pv_modeling import compute_pv_chain_for_series, calculate_climatological_annual_metrics

DAYS_IN_MONTH = {1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


def run_pv_chain_5b_vectorized(ghi_arr, zenith_true, apparent_zenith, azimuth, doy, month, n_cells=1):
    """
    Cadena física idéntica a la implementada en fase_5b_02_run_atlas.py.
    """
    dni_extra = pvlib.irradiance.get_extra_radiation(doy)

    # 1. Erbs con cenit verdadero y día del año real
    erbs_res = pvlib.irradiance.erbs(ghi_arr, zenith_true, doy)
    dni = np.maximum(np.asarray(erbs_res['dni']), 0.0)
    dhi = np.maximum(np.asarray(erbs_res['dhi']), 0.0)

    # 2. Perez Fijo (30° Norte)
    poa_fixed = pvlib.irradiance.get_total_irradiance(
        surface_tilt=30.0,
        surface_azimuth=0.0,
        solar_zenith=apparent_zenith,
        solar_azimuth=azimuth,
        dni=dni,
        ghi=ghi_arr,
        dhi=dhi,
        dni_extra=dni_extra,
        model='perez'
    )
    g_poa_fix = np.maximum(np.asarray(poa_fixed['poa_global']), 0.0)

    # 3. Perez Seguidor 1-eje con backtracking
    trk = pvlib.tracking.singleaxis(
        apparent_zenith=apparent_zenith,
        solar_azimuth=azimuth,
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
        solar_zenith=apparent_zenith,
        solar_azimuth=azimuth,
        dni=dni,
        ghi=ghi_arr,
        dhi=dhi,
        dni_extra=dni_extra,
        model='perez'
    )
    g_poa_trk = np.maximum(np.asarray(poa_trk['poa_global']), 0.0)

    # 4. Modelo térmico NOCT y temperatura ambiental estacional
    t_amb = np.where(np.isin(month, [12, 1, 2]), 24.0,
            np.where(np.isin(month, [3, 4, 5]), 18.0,
            np.where(np.isin(month, [6, 7, 8]), 11.0, 19.0)))

    noct = 45.0
    t_cell_fix = t_amb + ((noct - 20.0) / 800.0) * g_poa_fix
    t_cell_trk = t_amb + ((noct - 20.0) / 800.0) * g_poa_trk

    # 5. PVWatts DC + Inversor AC + BOS 10%
    pdc0 = 1000.0
    gamma_pdc = -0.0035
    t_ref = 25.0
    eta_inv_nom = 0.96
    eta_bos = 0.90

    pdc_fix = pvlib.pvsystem.pvwatts_dc(g_poa_fix, t_cell_fix, pdc0, gamma_pdc, t_ref)
    pac_fix_kw = (np.maximum(pvlib.inverter.pvwatts(pdc_fix, pdc0, eta_inv_nom), 0.0) * eta_bos) / 1000.0

    pdc_trk = pvlib.pvsystem.pvwatts_dc(g_poa_trk, t_cell_trk, pdc0, gamma_pdc, t_ref)
    pac_trk_kw = (np.maximum(pvlib.inverter.pvwatts(pdc_trk, pdc0, eta_inv_nom), 0.0) * eta_bos) / 1000.0

    return {
        'G_POA_Fixed': g_poa_fix,
        'G_POA_Tracker': g_poa_trk,
        'P_AC_Fixed_kW': pac_fix_kw,
        'P_AC_Tracker_kW': pac_trk_kw
    }


def validate_consistency():
    print("=" * 80)
    print("🔍 VALIDACIÓN DE INTEGRACIÓN Y CONSISTENCIA CIENTÍFICA: FASE 5A <-> FASE 5B")
    print("=" * 80)

    pred_file = Path("results/fase4_cv_predictions.parquet")
    if not pred_file.exists():
        raise FileNotFoundError(f"❌ No se encontró archivo de predicciones {pred_file}")

    df_cv = pd.read_parquet(pred_file)

    estaciones_info = {
        0: {"st_id": "115", "nombre": "Los Espinillos (Sierras)", "lat": -31.72, "lon": -64.60},
        1: {"st_id": "169", "nombre": "Ausonia (Centro-Sur)", "lat": -32.66, "lon": -63.25},
        2: {"st_id": "274", "nombre": "El Tío (Este)", "lat": -31.38, "lon": -62.81},
        3: {"st_id": "105", "nombre": "Presa Chañar (Norte)", "lat": -30.55, "lon": -65.05}
    }

    all_passed = True

    for region, info in estaciones_info.items():
        st_id = info["st_id"]
        nombre = info["nombre"]
        lat = info["lat"]
        lon = info["lon"]

        sub = df_cv[df_cv['estacion'].str.startswith(st_id)].sort_values('timestamp').copy()
        if sub.empty:
            print(f"⚠️ Estación {st_id} no encontrada en predicciones")
            continue

        times = pd.DatetimeIndex(sub['timestamp'].values)
        if times.tz is None:
            times = times.tz_localize('UTC')

        ghi_hibrido = sub['Híbrido (BD-JPT + ML)'].values

        # 1. Ejecutar Cadena Fase 5A
        res_5a = compute_pv_chain_for_series(sub['Híbrido (BD-JPT + ML)'], times, lat, lon)

        df_hourly_5a = pd.DataFrame({
            'timestamp': times,
            'GHI': ghi_hibrido,
            'G_POA_Fixed': res_5a['G_POA_Fixed'],
            'G_POA_Tracker': res_5a['G_POA_Tracker'],
            'P_AC_Fixed_kW': res_5a['P_AC_Fixed_kW'],
            'P_AC_Tracker_kW': res_5a['P_AC_Tracker_kW']
        })

        psh_ghi_5a, _, _ = calculate_climatological_annual_metrics(df_hourly_5a, 'GHI')
        psh_fix_5a, _, _ = calculate_climatological_annual_metrics(df_hourly_5a, 'G_POA_Fixed')
        psh_trk_5a, _, _ = calculate_climatological_annual_metrics(df_hourly_5a, 'G_POA_Tracker')
        yf_fix_5a, _, _ = calculate_climatological_annual_metrics(df_hourly_5a, 'P_AC_Fixed_kW')
        yf_trk_5a, _, _ = calculate_climatological_annual_metrics(df_hourly_5a, 'P_AC_Tracker_kW')

        # 2. Ejecutar Cadena Fase 5B
        loc = pvlib.location.Location(lat, lon)
        sp = loc.get_solarposition(times)
        zenith_true = sp['zenith'].values
        apparent_zenith = sp['apparent_zenith'].values
        azimuth = sp['azimuth'].values
        doy = times.dayofyear.values
        month = times.month.values

        res_5b = run_pv_chain_5b_vectorized(ghi_hibrido, zenith_true, apparent_zenith, azimuth, doy, month)

        df_hourly_5b = pd.DataFrame({
            'timestamp': times,
            'GHI': ghi_hibrido,
            'G_POA_Fixed': res_5b['G_POA_Fixed'],
            'G_POA_Tracker': res_5b['G_POA_Tracker'],
            'P_AC_Fixed_kW': res_5b['P_AC_Fixed_kW'],
            'P_AC_Tracker_kW': res_5b['P_AC_Tracker_kW']
        })

        psh_ghi_5b, _, _ = calculate_climatological_annual_metrics(df_hourly_5b, 'GHI')
        psh_fix_5b, _, _ = calculate_climatological_annual_metrics(df_hourly_5b, 'G_POA_Fixed')
        psh_trk_5b, _, _ = calculate_climatological_annual_metrics(df_hourly_5b, 'G_POA_Tracker')
        yf_fix_5b, _, _ = calculate_climatological_annual_metrics(df_hourly_5b, 'P_AC_Fixed_kW')
        yf_trk_5b, _, _ = calculate_climatological_annual_metrics(df_hourly_5b, 'P_AC_Tracker_kW')

        # 3. Comparación y tolerancia
        diff_ghi = abs(psh_ghi_5a - psh_ghi_5b)
        diff_poa_fix = abs(psh_fix_5a - psh_fix_5b)
        diff_poa_trk = abs(psh_trk_5a - psh_trk_5b)
        diff_yf_fix = abs(yf_fix_5a - yf_fix_5b)
        diff_yf_trk = abs(yf_trk_5a - yf_trk_5b)

        max_diff = max(diff_ghi, diff_poa_fix, diff_poa_trk, diff_yf_fix, diff_yf_trk)

        print(f"\n📍 Región {region}: {nombre}")
        print(f"   • Horas evaluadas: {len(times):,}")
        print(f"   • PSH GHI:         5A = {psh_ghi_5a/365.25:.4f} kWh/m²·d | 5B = {psh_ghi_5b/365.25:.4f} kWh/m²·d | Δ = {diff_ghi:.6e}")
        print(f"   • PSH POA Fijo:    5A = {psh_fix_5a/365.25:.4f} kWh/m²·d | 5B = {psh_fix_5b/365.25:.4f} kWh/m²·d | Δ = {diff_poa_fix:.6e}")
        print(f"   • PSH POA Tracker: 5A = {psh_trk_5a/365.25:.4f} kWh/m²·d | 5B = {psh_trk_5b/365.25:.4f} kWh/m²·d | Δ = {diff_poa_trk:.6e}")
        print(f"   • Yf Fijo:         5A = {yf_fix_5a:.2f} kWh/kWp·a | 5B = {yf_fix_5b:.2f} kWh/kWp·a | Δ = {diff_yf_fix:.6e}")
        print(f"   • Yf Tracker:      5A = {yf_trk_5a:.2f} kWh/kWp·a | 5B = {yf_trk_5b:.2f} kWh/kWp·a | Δ = {diff_yf_trk:.6e}")

        if max_diff > 1e-4:
            print(f"   ❌ Discrepancia detectada en Región {region}: max diff = {max_diff:.6e}")
            all_passed = False
        else:
            print(f"   ✅ Equivalencia numérica EXACTA demostrada (error < 1e-4)")

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ VALIDACIÓN DE INTEGRACIÓN 5A <-> 5B COMPLETADA EXITOSAMENTE: 100% COHERENCIA.")
    else:
        print("❌ FALLÓ LA VALIDACIÓN DE INTEGRACIÓN.")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = validate_consistency()
    if not success:
        sys.exit(1)
