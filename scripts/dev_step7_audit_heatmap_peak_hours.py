#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditoría Exhaustiva de Huso Horario: Heatmap Estación × Mes del Pico Solar (2018 - 2025)
========================================================================================
Construye la matriz completa de h_peak(estacion, mes) sobre los datos pre-calibración (QC raw)
para las 166 estaciones aprobadas de la red Davis.

Permite responder de forma terminante:
1. ¿Cuándo y en qué estaciones ocurrió la ruptura temporal de huso horario?
2. ¿Fue simultáneo o dependió del operador (APRHI vs Bioagroindustria)?
3. ¿Cuál es la transformación exacta requerida para corregir la capa de ingesta en Fase 2?
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT_DIR = Path(__file__).resolve().parent.parent
QC_FILE = ROOT_DIR / "data" / "processed" / "stations" / "davis_qc_hourly.parquet"
META_FILE = ROOT_DIR / "data" / "raw" / "aws" / "resumen_estaciones_sol_todas.csv"
RESULTS_DIR = ROOT_DIR / "results"
FIG_DIR = ROOT_DIR / "reportes" / "figures"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔬 AUDITORÍA PRE-CALIBRACIÓN: HEATMAP ESTACIÓN × MES DE PICO SOLAR (2018-2025)")
    print("=" * 80)

    # 1. Obtener las 166 estaciones aprobadas
    print("\n📂 1. Cargando metadatos de estaciones aprobadas...")
    meta_df = pd.read_csv(META_FILE)
    meta_df["queda_num"] = meta_df["queda"].astype(str).str.replace(",", ".").astype(float)
    approved_stations = sorted(list(set(meta_df[meta_df["queda_num"].isin([1.0, 0.5, 0.7, 0.2])]["estacion"])))
    print(f"   ✓ {len(approved_stations)} estaciones aprobadas seleccionadas.")

    # 2. Cargar datos horarios pre-calibración (davis_qc_hourly.parquet)
    print("\n📂 2. Cargando datos horarios pre-calibración (QC raw)...")
    df_qc = pd.read_parquet(QC_FILE)
    if "tiempo_utc" in df_qc.index.names:
        df_qc = df_qc.reset_index()
    df_qc["timestamp"] = pd.to_datetime(df_qc["tiempo_utc"], utc=True)
    df_qc = df_qc[df_qc["estacion"].isin(approved_stations)].copy()

    df_qc["year"] = df_qc["timestamp"].dt.year
    df_qc["month"] = df_qc["timestamp"].dt.month
    df_qc["hour"] = df_qc["timestamp"].dt.hour
    df_qc["year_month"] = df_qc["timestamp"].dt.to_period("M").astype(str)

    # Filtrar años de interés: 2018 a 2025
    df_qc = df_qc[(df_qc["year"] >= 2018) & (df_qc["year"] <= 2025)].copy()
    print(f"   ✓ {len(df_qc):,} observaciones diurnas/nocturnas en el período 2018-2025.")

    # 3. Generar lista exhaustiva de meses (2018-01 a 2025-12 = 96 meses)
    all_months = [f"{y}-{m:02d}" for y in range(2018, 2026) for m in range(1, 13)]

    # 4. Calcular el pico solar por estación y mes
    print("\n⏱️ 3. Calculando envolvente del 90% y pico horario por estación y mes...")
    
    # Agrupación eficiente: calculamos el cuantil 0.90 de radiación solar por (estacion, year_month, hour)
    hourly_envelopes = (
        df_qc[df_qc["solar_radiation"] > 5.0]
        .groupby(["estacion", "year_month", "hour"])["solar_radiation"]
        .agg(q90=lambda s: np.quantile(s, 0.90), count="count")
        .reset_index()
    )

    # Filtrar horas con muestra mínima
    hourly_envelopes = hourly_envelopes[hourly_envelopes["count"] >= 5].copy()

    # Encontrar el pico (hora de máxima irradiancia en la envolvente)
    idx_max = hourly_envelopes.groupby(["estacion", "year_month"])["q90"].idxmax()
    peaks = hourly_envelopes.loc[idx_max].copy()

    # Filtrar picos con irradiancia diurna genuina (q90 > 50 W/m²)
    peaks = peaks[peaks["q90"] > 50.0].copy()

    # Construir la matriz pivote (estaciones × meses)
    peak_matrix = peaks.pivot(index="estacion", columns="year_month", values="hour")
    
    # Reindexar para asegurar todas las 166 estaciones y los 96 meses
    peak_matrix = peak_matrix.reindex(index=approved_stations, columns=all_months)

    # Guardar matriz completa en CSV
    peak_matrix.to_csv(RESULTS_DIR / "davis_raw_peak_hour_matrix_2018_2025.csv")
    print(f"   ✓ Matriz guardada en results/davis_raw_peak_hour_matrix_2018_2025.csv ({peak_matrix.shape[0]} estaciones × {peak_matrix.shape[1]} meses).")

    # 5. Análisis Estadístico de Regímenes y Breakers
    print("\n" + "=" * 80)
    print("📊 4. ANÁLISIS DEL CAMBIO DE RÉGIMEN TEMPORAL POR AÑO Y MES")
    print("=" * 80)
    
    monthly_stats = []
    for ym in all_months:
        col = peak_matrix[ym].dropna()
        n_active = len(col)
        if n_active == 0:
            continue
        p16 = (col == 16).sum()
        p15 = (col == 15).sum()
        p14 = (col == 14).sum()
        p13 = (col == 13).sum()
        other = n_active - (p16 + p15 + p14 + p13)
        
        monthly_stats.append({
            "year_month": ym,
            "N_activas": n_active,
            "Pct_16_UTC": (p16 / n_active) * 100.0,
            "Pct_15_UTC": (p15 / n_active) * 100.0,
            "Pct_14_UTC": (p14 / n_active) * 100.0,
            "Pct_13_UTC": (p13 / n_active) * 100.0,
            "Pct_Otros": (other / n_active) * 100.0,
            "Mediana_Pico": col.median()
        })
    df_regime = pd.DataFrame(monthly_stats)
    
    # Mostrar tabla resumen para meses clave de 2019, 2020 y 2021
    key_months = [ym for ym in df_regime["year_month"] if ym.startswith("2019-") or ym.startswith("2020-") or ym.startswith("2021-")]
    print(df_regime[df_regime["year_month"].isin(key_months)][["year_month", "N_activas", "Pct_16_UTC", "Pct_14_UTC", "Pct_13_UTC", "Mediana_Pico"]].to_string(index=False))
    df_regime.to_csv(RESULTS_DIR / "davis_raw_peak_hour_regime_evolution.csv", index=False)

    # 6. Identificar fecha exacta de breakpoint por estación
    print("\n" + "=" * 80)
    print("🔍 5. ANÁLISIS DE BREAKPOINT INDIVIDUAL POR ESTACIÓN")
    print("=" * 80)
    
    station_breakpoints = []
    for st in approved_stations:
        s_series = peak_matrix.loc[st].dropna()
        if len(s_series) == 0:
            continue
        
        first_m = s_series.index[0]
        last_m = s_series.index[-1]
        
        # Detectar el primer mes donde se establece en 16 UTC (o 15-16 UTC)
        # y si antes estuvo en 13 o 14 UTC
        has_early_regime = any(s_series[m] in [13, 14] for m in s_series.index if m < "2020-11")
        has_utc_regime = any(s_series[m] in [16, 17] for m in s_series.index if m >= "2020-11")
        
        # Encontrar el mes exacto del salto
        jump_month = None
        for i in range(len(s_series) - 1):
            m1 = s_series.index[i]
            m2 = s_series.index[i+1]
            if s_series[m1] in [13, 14] and s_series[m2] in [16, 17]:
                jump_month = m2
                break
                
        station_breakpoints.append({
            "estacion": st,
            "inicio_datos": first_m,
            "fin_datos": last_m,
            "tuvo_regimen_13_14": has_early_regime,
            "tiene_regimen_16": has_utc_regime,
            "mes_del_salto": jump_month if jump_month else ("Siempre 16" if not has_early_regime and has_utc_regime else "Otro")
        })
    df_bp = pd.DataFrame(station_breakpoints)
    print(f"   • Estaciones con salto 13/14 -> 16 detectado: {(df_bp['tuvo_regimen_13_14'] & df_bp['tiene_regimen_16']).sum()}")
    print(f"   • Estaciones que comenzaron directamente en régimen 16 UTC: {(~df_bp['tuvo_regimen_13_14'] & df_bp['tiene_regimen_16']).sum()}")
    print("\nDistribución del mes exacto del salto:")
    print(df_bp["mes_del_salto"].value_counts())
    df_bp.to_csv(RESULTS_DIR / "davis_station_breakpoints.csv", index=False)

    # 7. GENERACIÓN DE FIGURAS DIAGNÓSTICAS
    print("\n🎨 6. Generando figuras diagnósticas de alta resolución...")
    
    # FIGURA 1: Heatmap Estación × Mes (2018-2025)
    fig, ax = plt.subplots(figsize=(16, 18), dpi=200)
    
    # Mapeo de colores discreto para horas de pico
    # 13: Local Solar Noon (Naranja Oscuro)
    # 14: Local Clock Noon (Naranja Claro)
    # 15: Transición (Verde Claro)
    # 16: UTC Solar Noon (Verde Oscuro)
    # 17: UTC Tardío (Azul)
    # Otros: Magenta
    cmap = mcolors.ListedColormap([
        "#d95f02", # 13: Naranja quemado
        "#fdae61", # 14: Ámbar / amarillo
        "#a6d96a", # 15: Verde lima
        "#1a9641", # 16: Verde oscuro intenso (UTC canónico)
        "#2b83ba", # 17: Azul
        "#7b3294", # Otros
    ])
    bounds = [12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 23.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Preparar array numérico para imshow
    # Reemplazar valores fuera de rango por 20 (categoría Otros)
    mat_vals = peak_matrix.values.copy()
    mat_display = np.full_like(mat_vals, np.nan, dtype=float)
    for r in range(mat_vals.shape[0]):
        for c in range(mat_vals.shape[1]):
            val = mat_vals[r, c]
            if pd.isna(val):
                continue
            if val in [13, 14, 15, 16, 17]:
                mat_display[r, c] = val
            else:
                mat_display[r, c] = 20.0 # Otros
                
    # Visualizar con fondo gris claro para valores faltantes
    current_cmap = cmap.copy()
    current_cmap.set_bad(color="#f0f0f0")
    
    im = ax.imshow(mat_display, aspect="auto", cmap=current_cmap, norm=norm, interpolation="nearest")
    
    # Eje X: Meses
    # Mostrar etiquetas cada 6 meses
    x_ticks = [i for i, m in enumerate(all_months) if m.endswith("-01") or m.endswith("-07")]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([all_months[i] for i in x_ticks], rotation=45, ha="right", fontsize=9, fontweight="bold")
    
    # Eje Y: Estaciones (mostrar cada 2 estaciones para legibilidad)
    y_ticks = list(range(0, len(approved_stations), 2))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([approved_stations[i] for i in y_ticks], fontsize=6)
    
    # Línea vertical roja destacando 2020-11
    idx_2020_11 = all_months.index("2020-11")
    ax.axvline(idx_2020_11 - 0.5, color="#d62728", linestyle="-", lw=2.5, label="Ruptura Global (Noviembre 2020)")
    
    # Líneas verticales delgadas para cada año
    for y in range(2019, 2026):
        idx_yr = all_months.index(f"{y}-01")
        ax.axvline(idx_yr - 0.5, color="black", linestyle="--", lw=0.8, alpha=0.5)
        
    ax.set_title("Auditoría Temporal Pre-Calibración: Hora del Pico Solar Diario por Estación y Mes (2018 - 2025)\nRed de Estaciones Automáticas Davis (Córdoba)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Año - Mes", fontsize=11, fontweight="bold")
    ax.set_ylabel("Estación Meteorológica (166 Estaciones Aprobadas)", fontsize=11, fontweight="bold")
    
    # Barra de color con etiquetas claras
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.06, fraction=0.03, aspect=30)
    cbar.set_ticks([13, 14, 15, 16, 17, 20])
    cbar.set_ticklabels([
        "13:00 UTC (Mediodía Solar Local ART)",
        "14:00 UTC (Mediodía Oficial ART)",
        "15:00 UTC (Oeste / Transición)",
        "16:00 UTC (Mediodía Solar Canónico UTC)",
        "17:00 UTC (Este / Tarde)",
        "Otros Horarios"
    ], fontsize=9, fontweight="bold")
    
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figura_heatmap_davis_peak_hours_raw.png")
    plt.close()
    print(f"   ✓ Heatmap guardado en {FIG_DIR / 'figura_heatmap_davis_peak_hours_raw.png'}")

    # FIGURA 2: Evolución Temporal del Régimen (% de estaciones por hora de pico)
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=200)
    x = range(len(df_regime))
    ax.plot(x, df_regime["Pct_16_UTC"], marker="o", color="#1a9641", lw=2.5, label="Pico a las 16:00 UTC (Régimen Canónico UTC)")
    ax.plot(x, df_regime["Pct_14_UTC"], marker="s", color="#fdae61", lw=2, label="Pico a las 14:00 UTC (Régimen Hora Local)")
    ax.plot(x, df_regime["Pct_13_UTC"], marker="^", color="#d95f02", lw=2, label="Pico a las 13:00 UTC (Régimen Hora Solar Local)")
    ax.plot(x, df_regime["Pct_15_UTC"], marker="d", color="#a6d96a", lw=1.5, linestyle="--", label="Pico a las 15:00 UTC")
    
    # Línea vertical del breakpoint
    ax.axvline(idx_2020_11, color="#d62728", linestyle="-", lw=2, label="Breakpoint Global (2020-11)")
    
    # Formato de ticks
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([all_months[i] for i in x_ticks], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Porcentaje de Estaciones Activas (%)", fontsize=11, fontweight="bold")
    ax.set_title("Evolución Temporal del Régimen de Huso Horario en la Red Davis (2018 - 2025)", fontsize=12, fontweight="bold")
    ax.set_ylim(-2, 102)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center left", fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figura_timeline_davis_peak_regimes.png")
    plt.close()
    print(f"   ✓ Línea temporal guardada en {FIG_DIR / 'figura_timeline_davis_peak_regimes.png'}")

    print("\n✅ AUDITORÍA DE HEATMAP COMPLETADA CON ÉXITO.")


if __name__ == "__main__":
    main()
