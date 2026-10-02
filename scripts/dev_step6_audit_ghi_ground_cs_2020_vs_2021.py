#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditoría Exhaustiva de Consistencia Interanual: GHI_ground vs GHI_cs (2020 vs 2021)
===================================================================================
Investiga el origen físico y metodológico de la anomalía de Kt > 1.0 (38% en 2021)
y el drift interanual entre el conjunto de entrenamiento (2020) y prueba (2021).

Analiza de forma independiente:
1. Distribuciones de Kt (2020 vs 2021)
2. Distribuciones de GHI_ground (2020 vs 2021)
3. Distribuciones de GHI_cs (2020 vs 2021)
4. Desglose estratificado:
   - Red Davis (166 estaciones automáticas de silicio)
   - Estación de Referencia Pilar CMP21 (termopila de cuarzo BSRN)
5. Análisis horario de perfiles diurnos y detección de desfases de huso horario en 2020.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
DAVIS_FILE = ROOT_DIR / "data" / "processed" / "stations" / "davis_calibrated_hourly_final.parquet"
PILAR_FILE = ROOT_DIR / "data" / "processed" / "stations" / "pilar_cmp21_hourly.parquet"
RESULTS_DIR = ROOT_DIR / "results"
FIG_DIR = ROOT_DIR / "reportes" / "figures"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔬 AUDITORÍA INTERANUAL: GHI_ground / GHI_cs (2020 vs 2021)")
    print("=" * 80)

    # 1. Cargar y procesar mediciones de Davis
    print("\n📂 Cargando mediciones de red Davis...")
    davis = pd.read_parquet(DAVIS_FILE)
    if "tiempo_utc" in davis.index.names:
        davis = davis.reset_index()
    davis["timestamp"] = pd.to_datetime(davis["tiempo_utc"], utc=True)
    davis["year"] = davis["timestamp"].dt.year
    davis["month"] = davis["timestamp"].dt.month
    davis["hour"] = davis["timestamp"].dt.hour
    davis = davis[davis["year"].isin([2020, 2021])].copy()
    davis = davis.dropna(subset=["solar_radiation", "ghi_cs"]).copy()
    davis = davis[davis["ghi_cs"] > 50.0].copy()
    davis["Kt"] = davis["solar_radiation"] / davis["ghi_cs"]
    davis["Kt_orig"] = davis["ghi_original"] / davis["ghi_cs"]

    # 2. Cargar y procesar Pilar CMP21
    print("📂 Cargando mediciones de Pilar CMP21...")
    pilar = pd.read_parquet(PILAR_FILE)
    pilar["timestamp"] = pd.to_datetime(pilar["timestamp"], utc=True)
    pilar["year"] = pilar["timestamp"].dt.year
    pilar["month"] = pilar["timestamp"].dt.month
    pilar["hour"] = pilar["timestamp"].dt.hour
    pilar = pilar[pilar["year"].isin([2020, 2021])].copy()

    loc_pilar = pvlib.location.Location(-31.667, -63.883, altitude=338)
    cs_p = loc_pilar.get_clearsky(pd.DatetimeIndex(pilar["timestamp"]))
    pilar["ghi_cs"] = cs_p["ghi"].values
    pilar = pilar.dropna(subset=["GHI", "ghi_cs"]).copy()
    pilar = pilar[pilar["ghi_cs"] > 50.0].copy()
    pilar["Kt"] = pilar["GHI"] / pilar["ghi_cs"]

    # --- PARTE A: COMPARATIVA GLOBAL 2020 vs 2021 (PILAR vs DAVIS) ---
    print("\n" + "=" * 80)
    print("📊 PARTE A: RESUMEN ESTADÍSTICO GLOBAL 2020 vs 2021")
    print("=" * 80)

    summary_rows = []
    for source_name, df, col_meas in [("Pilar CMP21 (Referencia)", pilar, "GHI"), ("Red Davis (Calibrada)", davis, "solar_radiation"), ("Red Davis (Raw / Original)", davis, "ghi_original")]:
        for y in [2020, 2021]:
            sub = df[df["year"] == y]
            kt = sub[col_meas] / sub["ghi_cs"]
            summary_rows.append({
                "Dataset": source_name,
                "Año": y,
                "N_horas": len(sub),
                "GHI_meas_mean": float(sub[col_meas].mean()),
                "GHI_meas_median": float(sub[col_meas].median()),
                "GHI_cs_mean": float(sub["ghi_cs"].mean()),
                "Kt_mean": float(kt.mean()),
                "Kt_median": float(kt.median()),
                "Kt_p25": float(kt.quantile(0.25)),
                "Kt_p75": float(kt.quantile(0.75)),
                "Frac_Kt_gt_1": float((kt > 1.0).mean()),
                "Frac_Kt_gt_1_15": float((kt > 1.15).mean()),
            })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary[["Dataset", "Año", "N_horas", "GHI_meas_mean", "GHI_cs_mean", "Kt_median", "Frac_Kt_gt_1", "Frac_Kt_gt_1_15"]].to_string(index=False))
    df_summary.to_csv(RESULTS_DIR / "fase4_dev_audit_summary_2020_vs_2021.csv", index=False)

    # --- PARTE B: ANÁLISIS MENSUAL DAVIS (DESGLOSE 1 a 12) ---
    print("\n" + "=" * 80)
    print("📊 PARTE B: EVOLUCIÓN MENSUAL EN RED DAVIS (GHI_meas vs GHI_cs vs Kt)")
    print("=" * 80)
    monthly_rows = []
    for m in range(1, 13):
        sub20 = davis[(davis["year"] == 2020) & (davis["month"] == m)]
        sub21 = davis[(davis["year"] == 2021) & (davis["month"] == m)]
        
        # Pico medio horario en UTC
        peak20 = sub20.groupby("hour")["solar_radiation"].mean().idxmax() if len(sub20) > 0 else np.nan
        peak21 = sub21.groupby("hour")["solar_radiation"].mean().idxmax() if len(sub21) > 0 else np.nan

        monthly_rows.append({
            "Mes": m,
            "N_2020": len(sub20),
            "N_2021": len(sub21),
            "PeakHour_2020": peak20,
            "PeakHour_2021": peak21,
            "GHI_meas_2020": sub20["solar_radiation"].mean() if len(sub20) > 0 else np.nan,
            "GHI_meas_2021": sub21["solar_radiation"].mean() if len(sub21) > 0 else np.nan,
            "GHI_orig_2020": sub20["ghi_original"].mean() if len(sub20) > 0 else np.nan,
            "GHI_orig_2021": sub21["ghi_original"].mean() if len(sub21) > 0 else np.nan,
            "GHI_cs_2020": sub20["ghi_cs"].mean() if len(sub20) > 0 else np.nan,
            "GHI_cs_2021": sub21["ghi_cs"].mean() if len(sub21) > 0 else np.nan,
            "Kt_med_2020": sub20["Kt"].median() if len(sub20) > 0 else np.nan,
            "Kt_med_2021": sub21["Kt"].median() if len(sub21) > 0 else np.nan,
            "Frac_gt_1_2020": (sub20["Kt"] > 1.0).mean() if len(sub20) > 0 else np.nan,
            "Frac_gt_1_2021": (sub21["Kt"] > 1.0).mean() if len(sub21) > 0 else np.nan,
        })
    df_monthly = pd.DataFrame(monthly_rows)
    print(df_monthly[["Mes", "PeakHour_2020", "PeakHour_2021", "GHI_meas_2020", "GHI_meas_2021", "Kt_med_2020", "Kt_med_2021", "Frac_gt_1_2020", "Frac_gt_1_2021"]].to_string(index=False))
    df_monthly.to_csv(RESULTS_DIR / "fase4_dev_audit_monthly_davis.csv", index=False)

    # --- PARTE C: ANÁLISIS HORARIO (EL SALTO TEMPORAL ENTRE OCTUBRE Y NOVIEMBRE 2020) ---
    print("\n" + "=" * 80)
    print("📊 PARTE C: AUDITORÍA HORARIA DAVIS ANTES Y DESPUÉS DE NOVIEMBRE 2020")
    print("=" * 80)
    davis["periodo"] = "2021 Completo"
    davis.loc[(davis["year"] == 2020) & (davis["month"] <= 10), "periodo"] = "2020 (Ene - Oct)"
    davis.loc[(davis["year"] == 2020) & (davis["month"] >= 11), "periodo"] = "2020 (Nov - Dic)"

    hourly_profile = davis.groupby(["periodo", "hour"])[["solar_radiation", "ghi_cs"]].mean().reset_index()
    hourly_pivot_meas = hourly_profile.pivot(index="hour", columns="periodo", values="solar_radiation")
    hourly_pivot_cs = hourly_profile.pivot(index="hour", columns="periodo", values="ghi_cs")
    print("\nPerfil diurno medio de GHI_meas (W/m²):")
    print(hourly_pivot_meas.to_string())

    # --- PARTE D: GENERACIÓN DE FIGURA DIAGNÓSTICA MULTI-PANEL ---
    print("\n🎨 Generando figura diagnóstica de alta resolución...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=200)

    # Panel (a): Distribución de Kt (KDE / Hist) 2020 vs 2021 en Davis
    ax = axes[0, 0]
    kt_d20 = davis.loc[davis["year"] == 2020, "Kt"].clip(0, 1.5)
    kt_d21 = davis.loc[davis["year"] == 2021, "Kt"].clip(0, 1.5)
    ax.hist(kt_d20, bins=60, density=True, alpha=0.5, color="#1f77b4", label=f"Davis 2020 (Mediana = {kt_d20.median():.2f})")
    ax.hist(kt_d21, bins=60, density=True, alpha=0.5, color="#d62728", label=f"Davis 2021 (Mediana = {kt_d21.median():.2f})")
    ax.axvline(1.0, color="black", linestyle="--", lw=1.2, label=r"$Kt = 1.0$ (Límite Cielo Claro)")
    ax.set_xlabel(r"Índice de Claridad $Kt = GHI_{\mathrm{ground}} / GHI_{cs}$", fontsize=11)
    ax.set_ylabel("Densidad de Probabilidad", fontsize=11)
    ax.set_title("(a) Desplazamiento Radical de la Distribución de $Kt$ en Red Davis", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    # Panel (b): Pilar CMP21: Distribución Estable 2020 vs 2021
    ax = axes[0, 1]
    kt_p20 = pilar.loc[pilar["year"] == 2020, "Kt"].clip(0, 1.5)
    kt_p21 = pilar.loc[pilar["year"] == 2021, "Kt"].clip(0, 1.5)
    ax.hist(kt_p20, bins=60, density=True, alpha=0.5, color="#1f77b4", label=f"Pilar 2020 (Mediana = {kt_p20.median():.2f})")
    ax.hist(kt_p21, bins=60, density=True, alpha=0.5, color="#2ca02c", label=f"Pilar 2021 (Mediana = {kt_p21.median():.2f})")
    ax.axvline(1.0, color="black", linestyle="--", lw=1.2, label=r"$Kt = 1.0$ (Límite Teórico)")
    ax.set_xlabel(r"Índice de Claridad $Kt = GHI_{\mathrm{ground}} / GHI_{cs}$", fontsize=11)
    ax.set_ylabel("Densidad de Probabilidad", fontsize=11)
    ax.set_title("(b) Estación Patrón Pilar CMP21: Distribución Invariante", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    # Panel (c): Perfiles Diurnos Horarios (Descubrimiento del desfase en 2020 Ene-Oct)
    ax = axes[1, 0]
    hours = hourly_pivot_meas.index
    ax.plot(hours, hourly_pivot_meas["2020 (Ene - Oct)"], marker="o", color="#d62728", lw=2, label="Davis 2020 Ene-Oct (Pico a las 14:00 UTC)")
    ax.plot(hours, hourly_pivot_meas["2020 (Nov - Dic)"], marker="s", color="#ff7f0e", lw=2, label="Davis 2020 Nov-Dic (Pico a las 16:00 UTC)")
    ax.plot(hours, hourly_pivot_meas["2021 Completo"], marker="^", color="#2ca02c", lw=2, label="Davis 2021 Completo (Pico a las 16:00 UTC)")
    ax.plot(hours, hourly_pivot_cs["2021 Completo"], linestyle=":", color="black", lw=1.5, label="GHI Cielo Claro Teórico (Mediodía 16:15 UTC)")
    ax.axvline(14, color="#d62728", linestyle="--", alpha=0.7)
    ax.axvline(16, color="#2ca02c", linestyle="--", alpha=0.7)
    ax.set_xlabel("Hora del Día (UTC)", fontsize=11)
    ax.set_ylabel(r"GHI Medio ($\mathrm{W/m}^2$)", fontsize=11)
    ax.set_title(r"(c) Detección de Desfase de Huso Horario en Davis (Salto Ene-Oct 2020)", fontsize=11, fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)

    # Panel (d): Evolución de Mediana de Kt y Fracción > 1.0 mes a mes
    ax = axes[1, 1]
    meses = df_monthly["Mes"]
    ax.plot(meses, df_monthly["Kt_med_2020"], marker="o", color="#1f77b4", lw=2, label=r"Mediana $Kt$ 2020 (Davis)")
    ax.plot(meses, df_monthly["Kt_med_2021"], marker="s", color="#d62728", lw=2, label=r"Mediana $Kt$ 2021 (Davis)")
    ax.plot(meses, df_monthly["Frac_gt_1_2021"], marker="^", color="#ff7f0e", linestyle="--", lw=1.5, label=r"Fracción $Kt > 1.0$ (2021)")
    ax.plot(meses, df_monthly["Frac_gt_1_2020"], marker="v", color="#17becf", linestyle="--", lw=1.5, label=r"Fracción $Kt > 1.0$ (2020)")
    ax.axhline(1.0, color="black", linestyle=":", lw=1)
    ax.set_xlabel("Mes del Año", fontsize=11)
    ax.set_ylabel(r"Valor de $Kt$ / Proporción", fontsize=11)
    ax.set_title(r"(d) Ruptura Estructural de $Kt$ entre 2020 y 2021", fontsize=11, fontweight="bold")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "figura_fase4_dev_audit_ground_cs_2020_vs_2021.png")
    plt.close()
    print("✓ Figura diagnóstica guardada en reportes/figures/figura_fase4_dev_audit_ground_cs_2020_vs_2021.png")

    print("\n✅ AUDITORÍA COMPLETADA EXITOSAMENTE.")


if __name__ == "__main__":
    main()
