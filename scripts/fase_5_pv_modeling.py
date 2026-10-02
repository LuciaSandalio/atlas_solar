#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 5A: Conversión GHI -> Potencial Fotovoltaico y Validación Energética Regional.
===================================================================================
Metodología Física y Rigor de Ingeniería (Informe 4 / Convenio ATLAS II):
1. Descomposición radiativa (Erbs et al., 1982) con cenit verdadero (solpos['zenith']).
2. Transposición anisotrópica a plano del módulo (Perez et al., 1990) con cenit aparente:
   - Configuración A: Fijo 30° orientado al Norte (azimut 0°).
   - Configuración B: Seguidor 1-eje horizontal (eje N-S, seguimiento E-O con backtracking).
3. Modelo térmico de celda (NOCT) y modelo paramétrico completo PVWatts (DC + Inversor AC + pérdidas BOS).
4. Integración temporal climatológica mensual-horaria robusta (inmune a días incompletos y vacíos).
5. Traducción del error radiométrico de Fase 4 a error de rendimiento energético (GHI Real vs Híbrido).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pvlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'grid.color': '#E0E0E0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.7
})

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("reportes/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Días promedio por mes considerando año bisiesto medio (365.25 / 12)
DAYS_IN_MONTH = {1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

def compute_pv_chain_for_series(ghi_series, times, lat, lon):
    """
    Ejecuta la cadena de descomposición (Erbs), transposición (Perez) y conversión (PVWatts)
    para una serie de GHI dada.
    """
    times_idx = pd.DatetimeIndex(times)
    if times_idx.tz is None:
        times_idx = times_idx.tz_localize('UTC')
    else:
        times_idx = times_idx.tz_convert('UTC')

    loc = pvlib.location.Location(lat, lon)
    solpos = loc.get_solarposition(times_idx)
    
    # 1. Ángulos solares: Cenit verdadero para Erbs, Cenit aparente para transposición y tracker
    true_zenith = solpos['zenith'].values
    apparent_zenith = solpos['apparent_zenith'].values
    azimuth = solpos['azimuth'].values
    
    # Radiación extraterrestre normal (usando vector numpy de día del año para evitar incompatibilidad con pandas 2.2)
    dni_extra = pvlib.irradiance.get_extra_radiation(times_idx.dayofyear.values)
    
    ghi = np.maximum(ghi_series.values, 0.0)
    
    # 2. Descomposición Radiativa (Erbs et al. 1982) con cenit astronómico verdadero
    erbs_res = pvlib.irradiance.erbs(ghi, true_zenith, times_idx)
    dni = np.maximum(erbs_res['dni'].values, 0.0)
    dhi = np.maximum(erbs_res['dhi'].values, 0.0)
    
    # 3. Transposición A: Plano Fijo (Tilt 30°, Azimut 0° Norte)
    poa_fixed = pvlib.irradiance.get_total_irradiance(
        surface_tilt=30.0,
        surface_azimuth=0.0,
        solar_zenith=apparent_zenith,
        solar_azimuth=azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        model='perez'
    )
    ghi_poa_fixed = np.maximum(np.asarray(poa_fixed['poa_global']), 0.0)
    
    # 4. Transposición B: Seguidor 1-eje (eje N-S, rotación E-O con backtracking)
    tracker = pvlib.tracking.singleaxis(
        apparent_zenith=apparent_zenith,
        solar_azimuth=azimuth,
        axis_tilt=0.0,
        axis_azimuth=0.0,
        max_angle=60.0,
        backtrack=True,
        gcr=0.35
    )
    surf_tilt_trk = np.asarray(tracker['surface_tilt'])
    surf_az_trk = np.asarray(tracker['surface_azimuth'])
    # Rellenar NaN nocturnos del tracker con posición horizontal de reposo
    surf_tilt_trk = np.nan_to_num(surf_tilt_trk, nan=0.0)
    surf_az_trk = np.nan_to_num(surf_az_trk, nan=0.0)
    
    poa_tracker = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surf_tilt_trk,
        surface_azimuth=surf_az_trk,
        solar_zenith=apparent_zenith,
        solar_azimuth=azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        model='perez'
    )
    ghi_poa_tracker = np.maximum(np.asarray(poa_tracker['poa_global']), 0.0)
    
    # 5. Modelo Térmico NOCT (45°C) con temperatura ambiente mensual de Córdoba
    month = times_idx.month
    t_ambient = np.where(np.isin(month, [12, 1, 2]), 24.0,
                np.where(np.isin(month, [3, 4, 5]), 18.0,
                np.where(np.isin(month, [6, 7, 8]), 11.0, 19.0)))
    
    noct = 45.0
    t_cell_fixed = t_ambient + ((noct - 20.0) / 800.0) * ghi_poa_fixed
    t_cell_tracker = t_ambient + ((noct - 20.0) / 800.0) * ghi_poa_tracker
    
    # 6. Conversión Paramétrica PVWatts (DC + Inversor AC + Pérdidas BOS)
    # Potencia nominal DC: 1000 W (1 kWp)
    pdc0 = 1000.0
    gamma_pdc = -0.0035  # -0.35 %/°C (Mono-PERC)
    
    p_dc_fixed = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=ghi_poa_fixed,
        temp_cell=t_cell_fixed,
        pdc0=pdc0,
        gamma_pdc=gamma_pdc,
        temp_ref=25.0
    )
    p_ac_fixed_raw = pvlib.inverter.pvwatts(pdc=p_dc_fixed, pdc0=pdc0, eta_inv_nom=0.96)
    
    p_dc_tracker = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=ghi_poa_tracker,
        temp_cell=t_cell_tracker,
        pdc0=pdc0,
        gamma_pdc=gamma_pdc,
        temp_ref=25.0
    )
    p_ac_tracker_raw = pvlib.inverter.pvwatts(pdc=p_dc_tracker, pdc0=pdc0, eta_inv_nom=0.96)
    
    # Pérdidas Balance of System (cableado, suciedad, mismatch): 10% adicional
    eta_bos = 0.90
    p_ac_fixed_net_kw = (np.maximum(p_ac_fixed_raw, 0.0) * eta_bos) / 1000.0
    p_ac_tracker_net_kw = (np.maximum(p_ac_tracker_raw, 0.0) * eta_bos) / 1000.0
    
    return {
        'DNI': dni,
        'DHI': dhi,
        'G_POA_Fixed': ghi_poa_fixed,
        'G_POA_Tracker': ghi_poa_tracker,
        'T_Cell_Fixed': t_cell_fixed,
        'P_AC_Fixed_kW': p_ac_fixed_net_kw,
        'P_AC_Tracker_kW': p_ac_tracker_net_kw
    }

def calculate_climatological_annual_metrics(df_hourly, col_name):
    """
    Integra de forma robusta la energía anual calculando el ciclo diurno medio
    horario para cada uno de los 12 meses del año, y ponderando por los días de cada mes.
    """
    df = df_hourly.copy()
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    
    # Matriz mensual-horaria (12 meses x 24 horas)
    hourly_m = df.groupby(['month', 'hour'])[col_name].mean().unstack(fill_value=0.0)
    for h in range(24):
        if h not in hourly_m.columns:
            hourly_m[h] = 0.0
    hourly_m = hourly_m[sorted(hourly_m.columns)]
    
    # Integral horaria diaria para cada mes (sum over 24h, Δt = 1.0 h)
    daily_m = hourly_m.sum(axis=1) * 1.0
    
    # Anualización ponderada por mes
    annual_val = sum(daily_m[m] * DAYS_IN_MONTH[m] for m in range(1, 13))
    daily_avg = annual_val / 365.25
    return annual_val, daily_avg, daily_m

def prepare_continuous_plot_df(sub_slice, lat, lon):
    """
    Construye un dataframe horario continuo para graficación:
    - Reindexa sobre una secuencia horaria regular continua (freq='1h').
    - Las horas nocturnas y los períodos sin medición permanecen como NaN,
      de modo que matplotlib interrumpe las líneas de manera limpia, eliminando
      las diagonales nocturnas artificiales y artefactos de sombreado.
    """
    if sub_slice.empty:
        return sub_slice
    t_min = sub_slice['timestamp'].min().floor('h')
    t_max = sub_slice['timestamp'].max().ceil('h')
    full_idx = pd.date_range(t_min, t_max, freq='1h', tz='UTC')
    
    plot_df = sub_slice.set_index('timestamp').reindex(full_idx).reset_index()
    plot_df.rename(columns={'index': 'timestamp'}, inplace=True)
    return plot_df

def main():
    print("☀️ Iniciando Fase 5A: Conversión GHI -> Potencial FV y Validación Regional...")
    pred_file = RESULTS_DIR / "fase4_cv_predictions.parquet"
    if not pred_file.exists():
        print(f"❌ Error: No se encontró {pred_file}")
        return
        
    print("📂 Cargando predicciones OOF de la Fase 4...")
    df_all = pd.read_parquet(pred_file)
    
    estaciones_info = {
        0: {"st_id": 115.0, "nombre": "Los Espinillos (Sierras)", "lat": -31.72, "lon": -64.60},
        1: {"st_id": 169.0, "nombre": "Ausonia (Centro-Sur)", "lat": -32.66, "lon": -63.25},
        2: {"st_id": 274.0, "nombre": "El Tío (Este)", "lat": -31.38, "lon": -62.81},
        3: {"st_id": 105.0, "nombre": "Presa Chañar (Norte)", "lat": -30.55, "lon": -65.05}
    }
    
    resumen_metricas = []
    processed_dict = {}
    
    for region, info in estaciones_info.items():
        st_id = info["st_id"]
        if "estacion" in df_all.columns:
            prefix = f"{int(st_id)}_"
            sub = df_all[df_all["estacion"].str.startswith(prefix)].copy()
        elif "station_id" in df_all.columns:
            sub = df_all[df_all["station_id"] == st_id].copy()
        else:
            sub = pd.DataFrame()
            
        if sub.empty:
            print(f"⚠️ Sin datos para estación {st_id}")
            continue
            
        # 1. Asegurar marcas temporales horarias únicas y ordenadas
        sub['timestamp'] = pd.to_datetime(sub['timestamp'], utc=True)
        sub['Season'] = sub['Season'].astype(str)
        season_map = sub.groupby('timestamp')['Season'].first()
        
        # Promediar duplicados exactamente en el mismo timestamp
        sub_hourly = sub.groupby('timestamp', as_index=False)[['Híbrido (BD-JPT + ML)', 'GHI_Real', 'ghi_cs', 'cosZ']].mean()
        sub_hourly['Season'] = sub_hourly['timestamp'].map(season_map)
        sub_hourly = sub_hourly.sort_values('timestamp').reset_index(drop=True)
        
        # Verificar paso temporal y asegurar Δt = 1.0 h
        time_diffs = sub_hourly['timestamp'].diff().dropna()
        dt_hours = time_diffs.dt.total_seconds().div(3600)
        dt_daylight = dt_hours[dt_hours < 5.0]  # aislar tramos diurnos continuos
        dt_median = dt_daylight.median() if not dt_daylight.empty else 1.0
        pct_exact_1h = (dt_daylight == 1.0).mean() * 100.0 if not dt_daylight.empty else 100.0
        
        print(f"\n⚡ Estación {info['nombre']} (Región {region}): {len(sub_hourly):,} horas diurnas procesadas.")
        print(f"   ✓ Verificación Δt: mediana diurna = {dt_median:.1f} h ({pct_exact_1h:.1f}% pasos diurnos exactamente 1.0 h).")
        print(f"   ✓ Top 3 transiciones dt: {dt_hours.value_counts().head(3).to_dict()}")
        
        # 2. Correr cadena PV para Modelo Híbrido
        pv_hyb = compute_pv_chain_for_series(sub_hourly['Híbrido (BD-JPT + ML)'], sub_hourly['timestamp'], info['lat'], info['lon'])
        for k, v in pv_hyb.items():
            sub_hourly[k] = v
            
        # 3. Correr cadena PV para Medición Real de Superficie (GHI Real)
        pv_real = compute_pv_chain_for_series(sub_hourly['GHI_Real'], sub_hourly['timestamp'], info['lat'], info['lon'])
        for k, v in pv_real.items():
            sub_hourly[f"{k}_Real"] = v
            
        processed_dict[region] = sub_hourly
        
        # 4. Integración Temporal Climatológica Mensual-Horaria
        ann_ghi_hyb, psh_ghi_hyb, _ = calculate_climatological_annual_metrics(sub_hourly, 'Híbrido (BD-JPT + ML)')
        ann_poa_fix_hyb, psh_poa_fix_hyb, _ = calculate_climatological_annual_metrics(sub_hourly, 'G_POA_Fixed')
        ann_poa_trk_hyb, psh_poa_trk_hyb, _ = calculate_climatological_annual_metrics(sub_hourly, 'G_POA_Tracker')
        
        ann_kwh_fix_hyb, _, _ = calculate_climatological_annual_metrics(sub_hourly, 'P_AC_Fixed_kW')
        ann_kwh_trk_hyb, _, _ = calculate_climatological_annual_metrics(sub_hourly, 'P_AC_Tracker_kW')
        
        # Métricas con GHI Real
        ann_ghi_real, psh_ghi_real, _ = calculate_climatological_annual_metrics(sub_hourly, 'GHI_Real')
        ann_kwh_fix_real, _, _ = calculate_climatological_annual_metrics(sub_hourly, 'P_AC_Fixed_kW_Real')
        ann_kwh_trk_real, _, _ = calculate_climatological_annual_metrics(sub_hourly, 'P_AC_Tracker_kW_Real')
        
        # Factores de Capacidad y Ganancias
        cf_fix_hyb = (ann_kwh_fix_hyb / (8760.0 * 1.0)) * 100.0
        cf_trk_hyb = (ann_kwh_trk_hyb / (8760.0 * 1.0)) * 100.0
        gain_trk_hyb = ((ann_kwh_trk_hyb - ann_kwh_fix_hyb) / ann_kwh_fix_hyb) * 100.0
        
        # Errores traducidos a Energía (% y kWh/kWp año)
        err_ghi_pct = ((ann_ghi_hyb - ann_ghi_real) / ann_ghi_real) * 100.0
        err_energy_fix_pct = ((ann_kwh_fix_hyb - ann_kwh_fix_real) / ann_kwh_fix_real) * 100.0
        err_energy_fix_kwh = ann_kwh_fix_hyb - ann_kwh_fix_real
        
        resumen_metricas.append({
            "Region": f"Región {region}",
            "Estacion": info["nombre"],
            "PSH_GHI (kWh/m2 d)": round(psh_ghi_hyb / 1000.0, 2),
            "PSH_Fijo (kWh/m2 d)": round(psh_poa_fix_hyb / 1000.0, 2),
            "PSH_Tracker (kWh/m2 d)": round(psh_poa_trk_hyb / 1000.0, 2),
            "Yf_Fijo_Hibrido (kWh/kWp a)": round(ann_kwh_fix_hyb, 1),
            "Yf_Fijo_Real (kWh/kWp a)": round(ann_kwh_fix_real, 1),
            "Error_Yf_Fijo (%)": round(err_energy_fix_pct, 2),
            "CF_Fijo (%)": round(cf_fix_hyb, 1),
            "Yf_Tracker (kWh/kWp a)": round(ann_kwh_trk_hyb, 1),
            "CF_Tracker (%)": round(cf_trk_hyb, 1),
            "Ganancia_Tracking (%)": round(gain_trk_hyb, 1)
        })
        
    df_metrics = pd.DataFrame(resumen_metricas)
    print("\n" + "="*100)
    print("TABLA DE POTENCIAL FOTOVOLTAICO Y TRADUCCIÓN DE ERROR ENERGÉTICO (FASE 5A)")
    print("="*100)
    print(df_metrics.to_string(index=False))
    print("="*100 + "\n")
    
    # Exportar Tabla LaTeX
    tex_file = FIGURES_DIR / "tabla_fase5a_potencial_fotovoltaico_regiones.tex"
    latex_table = df_metrics.to_latex(
        index=False,
        caption="Potencial fotovoltaico por región solar para sistemas fijos (30°) y seguidores en 1 eje (eje N-S, rotación E-O), incluyendo la traducción del error radiométrico a error de producción energética.",
        label="tab:potencial_fv_regional_fase5a"
    )
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"📄 Tabla LaTeX guardada en {tex_file}")
    
    # --- FIGURA 1: Descomposición Radiativa (Erbs et al.) ---
    print("🎨 Generando Figura 1: Descomposición Radiativa (Erbs)...")
    sample_reg_idx = 1 if 1 in processed_dict else next(iter(processed_dict.keys()), None)
    if sample_reg_idx is not None:
        sample_df = processed_dict[sample_reg_idx]
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
        
        # Verano
        verano_days = sample_df[sample_df['Season'] == 'Verano'].sort_values('timestamp')
        sub_v = pd.DataFrame()
        if not verano_days.empty:
            t_start_v = verano_days['timestamp'].min() + pd.Timedelta(days=12)
            t_end_v = t_start_v + pd.Timedelta(days=4) - pd.Timedelta(hours=2)
            raw_v = verano_days[(verano_days['timestamp'] >= t_start_v) & (verano_days['timestamp'] <= t_end_v)]
            sub_v = prepare_continuous_plot_df(raw_v, estaciones_info[sample_reg_idx]['lat'], estaciones_info[sample_reg_idx]['lon'])
            
            axes[0].plot(sub_v['timestamp'], sub_v['Híbrido (BD-JPT + ML)'], label="GHI Total (Modelo Híbrido)", color="#2C3E50", lw=2)
            axes[0].plot(sub_v['timestamp'], sub_v['DNI'], label="DNI (Directa Normal)", color="#E74C3C", lw=1.8)
            axes[0].plot(sub_v['timestamp'], sub_v['DHI'], label="DHI (Difusa Horizontal)", color="#3498DB", lw=1.5, linestyle="--")
            axes[0].set_title(f"Descomposición Radiativa en Verano (Erbs et al.) – {estaciones_info[sample_reg_idx]['nombre']}", fontsize=13, weight="bold")
            axes[0].set_ylabel("Irradiancia (W/m²)")
            axes[0].legend(loc="upper right")
            axes[0].grid(True)
            axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))
            
        # Invierno
        invierno_days = sample_df[sample_df['Season'] == 'Invierno'].sort_values('timestamp')
        sub_i = pd.DataFrame()
        if not invierno_days.empty:
            t_start_i = invierno_days['timestamp'].min() + pd.Timedelta(days=12)
            t_end_i = t_start_i + pd.Timedelta(days=4) - pd.Timedelta(hours=2)
            raw_i = invierno_days[(invierno_days['timestamp'] >= t_start_i) & (invierno_days['timestamp'] <= t_end_i)]
            sub_i = prepare_continuous_plot_df(raw_i, estaciones_info[sample_reg_idx]['lat'], estaciones_info[sample_reg_idx]['lon'])
            
            axes[1].plot(sub_i['timestamp'], sub_i['Híbrido (BD-JPT + ML)'], label="GHI Total (Modelo Híbrido)", color="#2C3E50", lw=2)
            axes[1].plot(sub_i['timestamp'], sub_i['DNI'], label="DNI (Directa Normal)", color="#E74C3C", lw=1.8)
            axes[1].plot(sub_i['timestamp'], sub_i['DHI'], label="DHI (Difusa Horizontal)", color="#3498DB", lw=1.5, linestyle="--")
            axes[1].set_title(f"Descomposición Radiativa en Invierno (Erbs et al.) – {estaciones_info[sample_reg_idx]['nombre']}", fontsize=13, weight="bold")
            axes[1].set_ylabel("Irradiancia (W/m²)")
            axes[1].legend(loc="upper right")
            axes[1].grid(True)
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))
            
        plt.tight_layout()
        fig1_path = FIGURES_DIR / "figura_fase5a_descomposicion_erbs.png"
        plt.savefig(fig1_path, dpi=200)
        plt.close()
        print(f"✅ Guardado: {fig1_path}")
        
        # --- FIGURA 2: Transposición POA (Fijo vs Tracker) ---
        print("🎨 Generando Figura 2: Transposición Plano Inclinado (Perez)...")
        if not sub_v.empty:
            fig, ax = plt.subplots(figsize=(13, 5), dpi=200)
            ax.plot(sub_v['timestamp'], sub_v['Híbrido (BD-JPT + ML)'], label="GHI Horizontal", color="#7F8C8D", lw=1.5, linestyle=":")
            ax.plot(sub_v['timestamp'], sub_v['G_POA_Fixed'], label="G_POA Fijo 30° (Norte)", color="#2980B9", lw=2.0)
            ax.plot(sub_v['timestamp'], sub_v['G_POA_Tracker'], label="G_POA Seguidor 1-eje (eje N-S, seguimiento E-O)", color="#E67E22", lw=2.2)
            valid_poa = sub_v['G_POA_Fixed'].notna() & sub_v['G_POA_Tracker'].notna()
            ax.fill_between(sub_v['timestamp'], sub_v['G_POA_Fixed'], sub_v['G_POA_Tracker'], where=valid_poa, color="#F39C12", alpha=0.2, label="Ganancia por Seguimiento Solar")
            ax.set_title("Efecto de la Transposición y Seguimiento Solar sobre la Irradiancia en el Plano (G_POA)\nModelo Perez Anisotrópico con Backtracking", fontsize=13, weight="bold")
            ax.set_ylabel("Irradiancia en Plano (W/m²)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))
            ax.legend(loc="upper right")
            ax.grid(True)
            plt.tight_layout()
            fig2_path = FIGURES_DIR / "figura_fase5a_transposicion_poa_comparativa.png"
            plt.savefig(fig2_path, dpi=200)
            plt.close()
            print(f"✅ Guardado: {fig2_path}")
            
        # --- FIGURA 3: Potencia Fotovoltaica Neta y Dinámica Térmica ---
        print("🎨 Generando Figura 3: Potencia Fotovoltaica Neta (PVWatts AC) y T_cell...")
        if not sub_v.empty:
            fig, ax1 = plt.subplots(figsize=(13, 5.5), dpi=200)
            color_p = '#27AE60'
            ax1.set_xlabel("Fecha y Hora (UTC)")
            ax1.set_ylabel("Potencia Neta AC (kW / kWp)", color=color_p)
            p1 = ax1.plot(sub_v['timestamp'], sub_v['P_AC_Fixed_kW'], color="#3498DB", lw=1.8, label="Potencia AC (Fijo 30°)")
            p2 = ax1.plot(sub_v['timestamp'], sub_v['P_AC_Tracker_kW'], color=color_p, lw=2.2, label="Potencia AC (Seguidor 1-eje)")
            ax1.tick_params(axis='y', labelcolor=color_p)
            ax1.set_ylim(0, 1.05)
            ax1.grid(True)
            
            ax2 = ax1.twinx()
            color_t = '#E74C3C'
            ax2.set_ylabel("Temperatura de Celda (°C)", color=color_t)
            p3 = ax2.plot(sub_v['timestamp'], sub_v['T_Cell_Fixed'], color=color_t, lw=1.5, linestyle="--", label="T_Celda Fijo (°C)")
            ax2.tick_params(axis='y', labelcolor=color_t)
            ax2.set_ylim(15, 65)
            
            lines = p1 + p2 + p3
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))
            ax1.set_title("Curvas de Generación Neta AC y Penalización Térmica de Celda (PVWatts + NOCT)", fontsize=13, weight="bold")
            plt.tight_layout()
            fig3_path = FIGURES_DIR / "figura_fase5a_potencia_fotovoltaica_diaria.png"
            plt.savefig(fig3_path, dpi=200)
            plt.close()
            print(f"✅ Guardado: {fig3_path}")
            
    # --- FIGURA 4: Rendimiento Específico Comparativo y Validación Energética ---
    print("🎨 Generando Figura 4: Rendimiento Específico y Traducción de Error Energético...")
    if resumen_metricas:
        fig, (ax_bar, ax_err) = plt.subplots(1, 2, figsize=(15.5, 5.5), dpi=200, gridspec_kw={'width_ratios': [1.8, 1.2]})
        
        regions = [m["Region"] for m in resumen_metricas]
        y_fix = [m["Yf_Fijo_Hibrido (kWh/kWp a)"] for m in resumen_metricas]
        y_trk = [m["Yf_Tracker (kWh/kWp a)"] for m in resumen_metricas]
        gains = [m["Ganancia_Tracking (%)"] for m in resumen_metricas]
        err_pct = [m["Error_Yf_Fijo (%)"] for m in resumen_metricas]
        
        x = np.arange(len(regions))
        width = 0.35
        
        ax_bar.bar(x - width/2, y_fix, width, label='Fijo 30° Norte', color='#2980B9', alpha=0.9)
        ax_bar.bar(x + width/2, y_trk, width, label='Seguidor 1-eje (eje N-S, seguimiento E-O)', color='#E67E22', alpha=0.9)
        
        for i, g in enumerate(gains):
            ax_bar.annotate(f"+{g:.1f}%", (x[i] + width/2, y_trk[i] + 25), ha='center', va='bottom', fontsize=10, weight='bold', color='#D35400')
            
        ax_bar.set_ylabel("Rendimiento Específico Anualizado (kWh / kWp · año)")
        ax_bar.set_title("Rendimiento Específico Anualizado por Región Solar\nSistema fijo 30° vs. seguimiento a un eje (Provincia de Córdoba)", fontsize=12, weight="bold")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f"{m['Region']}\n{m['Estacion']}" for m in resumen_metricas], fontsize=9)
        ax_bar.set_ylim(0, max(y_trk) * 1.15)
        ax_bar.legend(loc="lower right")
        ax_bar.grid(True, axis='y', linestyle=':')
        
        # Gráfico de Propagación de Error Energético Híbrido vs Real
        colors_err = ['#27AE60' if abs(e) <= 3.0 else '#E67E22' for e in err_pct]
        ax_err.bar(x, err_pct, width=0.5, color=colors_err, alpha=0.85)
        ax_err.axhline(0, color='black', lw=1)
        ax_err.set_ylabel("Desvío del rendimiento modelado\nrespecto de GHI observado (%)", fontsize=11)
        ax_err.set_title("Propagación del error de GHI al\nrendimiento fotovoltaico modelado", fontsize=12, weight="bold")
        ax_err.set_xticks(x)
        ax_err.set_xticklabels([m['Region'] for m in resumen_metricas], fontsize=10)
        ax_err.grid(True, axis='y', linestyle=':')
        for i, e in enumerate(err_pct):
            pos_y = e + (0.3 if e >= 0 else -0.7)
            ax_err.annotate(f"{e:+.2f}%", (x[i], pos_y), ha='center', va='bottom', fontsize=10, weight='bold')
            
        plt.tight_layout(w_pad=3.0)
        fig4_path = FIGURES_DIR / "figura_fase5a_metricas_regionales_pv.png"
        plt.savefig(fig4_path, dpi=200)
        plt.close()
        print(f"✅ Guardado: {fig4_path}")
        
    print("\n🎉 ¡Fase 5A completada con éxito! Todas las correcciones metodológicas fueron aplicadas.")

if __name__ == "__main__":
    main()
