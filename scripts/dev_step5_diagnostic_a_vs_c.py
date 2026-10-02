#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnóstico Fino de Sesgo y Error: Opción A vs Opción C
======================================================
Desglosa el error fuera de muestra (Test 2021 OOS) en función de:
1. Índice de Claridad terrestre (Kt)
2. Mes del año (1 a 8)
3. Ángulo Cenital Solar (cos Z)
4. Radiación de Cielo Claro (GHI_cs)
5. Estación Patrón Pilar CMP21

Objetivo:
Verificar si el sesgo (MBE) en cielo claro y a lo largo de los meses
es común a ambas opciones (explicado por el drift interanual 2020->2021)
o si depende de la elección de sincronización satelital.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from models.bdjpt import BDJPTWrapper, HybridBDJPT_HGB

STATION_MAP_FILE = ROOT_DIR / "data" / "ready" / "station_to_grid_2km.parquet"
SAT_2KM_FILE = ROOT_DIR / "data" / "ready" / "b0_model" / "station_features_B0_2km_dev.parquet"
DAVIS_FILE = ROOT_DIR / "data" / "processed" / "stations" / "davis_calibrated_hourly_final.parquet"
PILAR_FILE = ROOT_DIR / "data" / "processed" / "stations" / "pilar_cmp21_hourly.parquet"

RESULTS_DIR = ROOT_DIR / "results"
FIG_DIR = ROOT_DIR / "reportes" / "figures"
ARTIFACT_DIR = Path("reportes/figures")


def calc_metrics_series(y_true, y_est):
    res = y_est - y_true
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae = float(np.mean(np.abs(res)))
    mbe = float(np.mean(res))
    mean_t = float(np.mean(y_true))
    r2 = float(1.0 - (np.sum(res ** 2) / np.sum((y_true - mean_t) ** 2))) if np.sum((y_true - mean_t) ** 2) > 0 else np.nan
    return {"N": len(y_true), "RMSE": rmse, "MAE": mae, "MBE": mbe, "R2": r2, "Mean_True": mean_t}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔬 DIAGNÓSTICO FINO DE ERROR Y SESGO: OPCIÓN A vs OPCIÓN C (TEST 2021)")
    print("=" * 80)

    # 1. Cargar Mapeo de Estaciones
    df_map = pd.read_parquet(STATION_MAP_FILE)
    st_to_cell = dict(zip(df_map["station_id"], df_map["cell_id_2km"]))
    st_to_reg = dict(zip(df_map["station_id"], df_map["region"]))

    # 2. Cargar Terrestres (Davis + Pilar)
    print("\n📂 Cargando mediciones terrestres...")
    davis = pd.read_parquet(DAVIS_FILE)
    if "tiempo_utc" in davis.index.names:
        davis = davis.reset_index()
    davis["timestamp"] = pd.to_datetime(davis["tiempo_utc"], utc=True)
    davis = davis[(davis["timestamp"] >= "2020-01-01") & (davis["timestamp"] <= "2021-08-03")].copy()
    davis["station_id"] = davis["estacion"].astype(str)
    davis["cell_id"] = davis["station_id"].map(st_to_cell)
    davis["region"] = davis["station_id"].map(st_to_reg)
    davis["target_GHI"] = davis["solar_radiation"]
    davis["is_davis"] = True

    pilar = pd.read_parquet(PILAR_FILE)
    pilar["timestamp"] = pd.to_datetime(pilar["timestamp"], utc=True)
    pilar = pilar[(pilar["timestamp"] >= "2020-01-01") & (pilar["timestamp"] <= "2021-08-03")].copy()
    pilar["station_id"] = "PILAR_CMP21"
    pilar["cell_id"] = st_to_cell["PILAR_CMP21"]
    pilar["region"] = st_to_reg["PILAR_CMP21"]
    pilar["target_GHI"] = pilar["GHI"]
    pilar["is_davis"] = False

    times_pilar = pd.DatetimeIndex(pilar["timestamp"])
    loc_pilar = pvlib.location.Location(-31.667, -63.883)
    cs_pilar = loc_pilar.get_clearsky(times_pilar)
    pilar["ghi_cs"] = cs_pilar["ghi"].values

    cols_g = ["station_id", "cell_id", "region", "timestamp", "target_GHI", "ghi_cs", "is_davis"]
    ground = pd.concat([davis[cols_g], pilar[cols_g]], ignore_index=True)
    del davis, pilar

    ground = ground.dropna(subset=["target_GHI", "ghi_cs", "cell_id"])
    ground = ground[ground["ghi_cs"] > 20.0].copy()
    ground["Kt_ground"] = (ground["target_GHI"] / ground["ghi_cs"]).clip(0.0, 1.15)
    ground["day_of_year"] = ground["timestamp"].dt.dayofyear
    ground["month"] = ground["timestamp"].dt.month
    print(f"✓ Registros terrestres diurnos listos: {len(ground):,}")

    # 3. Satélite 2 km Canónico (A y C)
    print("\n📂 Cargando y preparando sincronizaciones A y C...")
    sat = pd.read_parquet(SAT_2KM_FILE)
    sat["timestamp"] = pd.to_datetime(sat["timestamp_sat"], utc=True)

    # Opción A
    sat_A = sat[["cell_id", "timestamp", "reflectance", "B0", "Fr", "cosZ"]].copy()

    # Opción C
    sat_sorted = sat.sort_values(["cell_id", "timestamp"]).copy()
    prev_t = sat_sorted.groupby("cell_id")["timestamp"].shift(1)
    consec = (sat_sorted["timestamp"] - prev_t) == pd.Timedelta(hours=1)
    sat_C = sat_sorted.copy()
    for col in ["reflectance", "B0", "Fr", "cosZ"]:
        prev = sat_sorted.groupby("cell_id")[col].shift(1)
        sat_C[col] = np.where(consec, (sat_C[col] + prev) / 2.0, np.nan)
    sat_C["timestamp"] = sat_C["timestamp"] - pd.Timedelta(hours=1)
    sat_C = sat_C.dropna(subset=["reflectance", "B0", "Fr", "cosZ"]).copy()

    mA = pd.merge(ground, sat_A, on=["cell_id", "timestamp"], how="inner")
    mC = pd.merge(ground, sat_C, on=["cell_id", "timestamp"], how="inner")

    # Intersección común estricta
    keys_A = set(zip(mA["station_id"], mA["timestamp"]))
    keys_C = set(zip(mC["station_id"], mC["timestamp"]))
    common = keys_A.intersection(keys_C)
    print(f"✓ Pares comunes válidos A ∩ C: {len(common):,}")

    mask_cA = [k in common for k in zip(mA["station_id"], mA["timestamp"])]
    mask_cC = [k in common for k in zip(mC["station_id"], mC["timestamp"])]
    df_A = mA[mask_cA].sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    df_C = mC[mask_cC].sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    feat_cols = ["reflectance", "B0", "Fr", "cosZ", "day_of_year"]

    # 4. Entrenar HybridBDJPT_HGB sobre Train (2020) para A y C
    train_mask = df_A["timestamp"] < "2021-01-01"
    test_mask = df_A["timestamp"] >= "2021-01-01"

    print("\n🚀 Entrenando HybridBDJPT_HGB en 2020 para Opción A y Opción C...")
    # Opción A
    mod_A = HybridBDJPT_HGB(max_iter=150, max_depth=12, random_state=42)
    mod_A.fit(
        df_A.loc[train_mask, feat_cols].values,
        df_A.loc[train_mask, "Kt_ground"].values,
        df_A.loc[train_mask, "target_GHI"].values,
        df_A.loc[train_mask, "ghi_cs"].values
    )

    # Opción C
    mod_C = HybridBDJPT_HGB(max_iter=150, max_depth=12, random_state=42)
    mod_C.fit(
        df_C.loc[train_mask, feat_cols].values,
        df_C.loc[train_mask, "Kt_ground"].values,
        df_C.loc[train_mask, "target_GHI"].values,
        df_C.loc[train_mask, "ghi_cs"].values
    )
    print("✓ Modelos ajustados exitosamente.")

    # 4.5 Validación Cruzada Espacial (5-Fold GroupKFold) sobre Train (2020)
    print("\n🔄 Evaluando Validación Cruzada Espacial (5-Fold GroupKFold) sobre Train 2020...")
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    train_A = df_A[train_mask].reset_index(drop=True)
    train_C = df_C[train_mask].reset_index(drop=True)
    
    cv_A_res, cv_C_res = [], []
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_A, groups=train_A['station_id'])):
        # Model A
        mod_A_cv = HybridBDJPT_HGB(max_iter=100, max_depth=10, random_state=42)
        mod_A_cv.fit(train_A.loc[trn_idx, feat_cols].values, train_A.loc[trn_idx, "Kt_ground"].values, train_A.loc[trn_idx, "target_GHI"].values, train_A.loc[trn_idx, "ghi_cs"].values)
        pred_A_cv = mod_A_cv.predict(train_A.loc[val_idx, feat_cols].values, train_A.loc[val_idx, "ghi_cs"].values) * train_A.loc[val_idx, "ghi_cs"].values
        cv_A_res.append(calc_metrics_series(train_A.loc[val_idx, "target_GHI"].values, pred_A_cv))
        
        # Model C
        mod_C_cv = HybridBDJPT_HGB(max_iter=100, max_depth=10, random_state=42)
        mod_C_cv.fit(train_C.loc[trn_idx, feat_cols].values, train_C.loc[trn_idx, "Kt_ground"].values, train_C.loc[trn_idx, "target_GHI"].values, train_C.loc[trn_idx, "ghi_cs"].values)
        pred_C_cv = mod_C_cv.predict(train_C.loc[val_idx, feat_cols].values, train_C.loc[val_idx, "ghi_cs"].values) * train_C.loc[val_idx, "ghi_cs"].values
        cv_C_res.append(calc_metrics_series(train_C.loc[val_idx, "target_GHI"].values, pred_C_cv))
        
    print(f"   CV Espacial Opción A: RMSE={np.mean([r['RMSE'] for r in cv_A_res]):.2f} W/m², MAE={np.mean([r['MAE'] for r in cv_A_res]):.2f}, MBE={np.mean([r['MBE'] for r in cv_A_res]):.2f}, R2={np.mean([r['R2'] for r in cv_A_res]):.3f}")
    print(f"   CV Espacial Opción C: RMSE={np.mean([r['RMSE'] for r in cv_C_res]):.2f} W/m², MAE={np.mean([r['MAE'] for r in cv_C_res]):.2f}, MBE={np.mean([r['MBE'] for r in cv_C_res]):.2f}, R2={np.mean([r['R2'] for r in cv_C_res]):.3f}")

    # 5. Inferencia sobre Test 2021 (OOS)
    df_te = df_A[test_mask].copy() # datos de tierra comunes
    pred_kt_A = mod_A.predict(df_A.loc[test_mask, feat_cols].values, df_te["ghi_cs"].values)
    pred_kt_C = mod_C.predict(df_C.loc[test_mask, feat_cols].values, df_te["ghi_cs"].values)

    df_te["pred_GHI_A"] = pred_kt_A * df_te["ghi_cs"].values
    df_te["pred_GHI_C"] = pred_kt_C * df_te["ghi_cs"].values
    df_te["error_A"] = df_te["pred_GHI_A"] - df_te["target_GHI"]
    df_te["error_C"] = df_te["pred_GHI_C"] - df_te["target_GHI"]

    # RESUMEN GLOBAL TEST 2021
    print("\n" + "=" * 80)
    print("🏆 RESUMEN GENERAL FUERA DE MUESTRA (TEST 2021 COMPLETO: ENE-AGO)")
    print("=" * 80)
    glob_A = calc_metrics_series(df_te["target_GHI"].values, df_te["pred_GHI_A"].values)
    glob_C = calc_metrics_series(df_te["target_GHI"].values, df_te["pred_GHI_C"].values)
    print(f"Global Test 2021 (N={glob_A['N']:,}):")
    print(f"   • Opción A (t):           RMSE={glob_A['RMSE']:.2f} W/m² | MAE={glob_A['MAE']:.2f} | MBE={glob_A['MBE']:.2f} | R²={glob_A['R2']:.4f}")
    print(f"   • Opción C (mean t, t+1): RMSE={glob_C['RMSE']:.2f} W/m² | MAE={glob_C['MAE']:.2f} | MBE={glob_C['MBE']:.2f} | R²={glob_C['R2']:.4f}")

    # Pilar CMP21
    pilar_te = df_te[~df_te["is_davis"]]
    pil_A = calc_metrics_series(pilar_te["target_GHI"].values, pilar_te["pred_GHI_A"].values)
    pil_C = calc_metrics_series(pilar_te["target_GHI"].values, pilar_te["pred_GHI_C"].values)
    print(f"\nPilar CMP21 Patrón (N={pil_A['N']:,}):")
    print(f"   • Opción A (t):           RMSE={pil_A['RMSE']:.2f} W/m² | MAE={pil_A['MAE']:.2f} | MBE={pil_A['MBE']:.2f} | R²={pil_A['R2']:.4f}")
    print(f"   • Opción C (mean t, t+1): RMSE={pil_C['RMSE']:.2f} W/m² | MAE={pil_C['MAE']:.2f} | MBE={pil_C['MBE']:.2f} | R²={pil_C['R2']:.4f}")

    # Horas Nubladas (Kt < 0.65) Global y Pilar
    nub_te = df_te[df_te["Kt_ground"] < 0.65]
    nub_A = calc_metrics_series(nub_te["target_GHI"].values, nub_te["pred_GHI_A"].values)
    nub_C = calc_metrics_series(nub_te["target_GHI"].values, nub_te["pred_GHI_C"].values)
    print(f"\nHoras Nubladas Global (Kt < 0.65, N={nub_A['N']:,}):")
    print(f"   • Opción A (t):           RMSE={nub_A['RMSE']:.2f} W/m² | MAE={nub_A['MAE']:.2f} | MBE={nub_A['MBE']:.2f} | R²={nub_A['R2']:.4f}")
    print(f"   • Opción C (mean t, t+1): RMSE={nub_C['RMSE']:.2f} W/m² | MAE={nub_C['MAE']:.2f} | MBE={nub_C['MBE']:.2f} | R²={nub_C['R2']:.4f}")

    nub_pil = pilar_te[pilar_te["Kt_ground"] < 0.65]
    np_A = calc_metrics_series(nub_pil["target_GHI"].values, nub_pil["pred_GHI_A"].values)
    np_C = calc_metrics_series(nub_pil["target_GHI"].values, nub_pil["pred_GHI_C"].values)
    print(f"\nPilar CMP21 Horas Nubladas (Kt < 0.65, N={np_A['N']:,}):")
    print(f"   • Opción A (t):           RMSE={np_A['RMSE']:.2f} W/m² | MAE={np_A['MAE']:.2f} | MBE={np_A['MBE']:.2f} | R²={np_A['R2']:.4f}")
    print(f"   • Opción C (mean t, t+1): RMSE={np_C['RMSE']:.2f} W/m² | MAE={np_C['MAE']:.2f} | MBE={np_C['MBE']:.2f} | R²={np_C['R2']:.4f}")

    # 6. Desgloses Diagnósticos Específicos
    print("\n" + "=" * 80)
    print("📊 1. ANÁLISIS DE SESGO Y ERROR POR RANGOS DE Kt (ÍNDICE DE CLARIDAD)")
    print("=" * 80)
    kt_bins = [0.0, 0.20, 0.40, 0.60, 0.80, 1.00, 1.20]
    kt_labels = ["0.0-0.2 (Muy Oscuro/Tormenta)", "0.2-0.4 (Nublado Severo)", "0.4-0.6 (Nublado Moderado)", "0.6-0.8 (Cielo Claro)", "0.8-1.0 (Despejado Pleno)", ">1.0 (Sobre-irradiancia)"]
    df_te["kt_bin"] = pd.cut(df_te["Kt_ground"], bins=kt_bins, labels=kt_labels, right=False)

    kt_rows = []
    for b in kt_labels:
        sub = df_te[df_te["kt_bin"] == b]
        if len(sub) == 0:
            continue
        mA_res = calc_metrics_series(sub["target_GHI"].values, sub["pred_GHI_A"].values)
        mC_res = calc_metrics_series(sub["target_GHI"].values, sub["pred_GHI_C"].values)
        kt_rows.append({
            "Rango_Kt": b,
            "N": len(sub),
            "GHI_Medio": mA_res["Mean_True"],
            "MBE_A": mA_res["MBE"],
            "MBE_C": mC_res["MBE"],
            "Diff_MBE_C_minus_A": mC_res["MBE"] - mA_res["MBE"],
            "RMSE_A": mA_res["RMSE"],
            "RMSE_C": mC_res["RMSE"],
            "MAE_A": mA_res["MAE"],
            "MAE_C": mC_res["MAE"],
        })
    df_kt_diag = pd.DataFrame(kt_rows)
    print(df_kt_diag[["Rango_Kt", "N", "GHI_Medio", "MBE_A", "MBE_C", "RMSE_A", "RMSE_C"]].to_string(index=False))
    df_kt_diag.to_csv(RESULTS_DIR / "fase4_dev_diagnostic_a_vs_c_kt.csv", index=False)

    print("\n" + "=" * 80)
    print("📊 2. ANÁLISIS DE SESGO Y ERROR POR MES DEL AÑO (DRIFT TEMPORAL 2021)")
    print("=" * 80)
    mes_names = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto"}
    mes_rows = []
    for m_num in sorted(df_te["month"].unique()):
        sub = df_te[df_te["month"] == m_num]
        mA_res = calc_metrics_series(sub["target_GHI"].values, sub["pred_GHI_A"].values)
        mC_res = calc_metrics_series(sub["target_GHI"].values, sub["pred_GHI_C"].values)
        mes_rows.append({
            "Mes_Num": m_num,
            "Mes": mes_names.get(m_num, str(m_num)),
            "N": len(sub),
            "GHI_Medio": mA_res["Mean_True"],
            "MBE_A": mA_res["MBE"],
            "MBE_C": mC_res["MBE"],
            "Diff_MBE": mC_res["MBE"] - mA_res["MBE"],
            "RMSE_A": mA_res["RMSE"],
            "RMSE_C": mC_res["RMSE"],
            "R2_A": mA_res["R2"],
            "R2_C": mC_res["R2"]
        })
    df_mes_diag = pd.DataFrame(mes_rows)
    print(df_mes_diag[["Mes", "N", "GHI_Medio", "MBE_A", "MBE_C", "Diff_MBE", "RMSE_A", "RMSE_C", "R2_A", "R2_C"]].to_string(index=False))
    df_mes_diag.to_csv(RESULTS_DIR / "fase4_dev_diagnostic_a_vs_c_mes.csv", index=False)

    print("\n" + "=" * 80)
    print("📊 3. ANÁLISIS DE SESGO Y ERROR POR ÁNGULO CENITAL (cos Z)")
    print("=" * 80)
    cosz_bins = [0.0, 0.25, 0.50, 0.75, 1.00]
    cosz_labels = ["0.00-0.25 (Cerca del Horizonte)", "0.25-0.50 (Media Mañana/Tarde)", "0.50-0.75 (Cenital Medio)", "0.75-1.00 (Mediodía Solar)"]
    df_te["cosz_bin"] = pd.cut(df_te["cosZ"], bins=cosz_bins, labels=cosz_labels, right=False)

    cosz_rows = []
    for b in cosz_labels:
        sub = df_te[df_te["cosz_bin"] == b]
        if len(sub) == 0:
            continue
        mA_res = calc_metrics_series(sub["target_GHI"].values, sub["pred_GHI_A"].values)
        mC_res = calc_metrics_series(sub["target_GHI"].values, sub["pred_GHI_C"].values)
        cosz_rows.append({
            "Rango_cosZ": b,
            "N": len(sub),
            "MBE_A": mA_res["MBE"],
            "MBE_C": mC_res["MBE"],
            "RMSE_A": mA_res["RMSE"],
            "RMSE_C": mC_res["RMSE"]
        })
    df_cosz_diag = pd.DataFrame(cosz_rows)
    print(df_cosz_diag.to_string(index=False))
    df_cosz_diag.to_csv(RESULTS_DIR / "fase4_dev_diagnostic_a_vs_c_cosZ.csv", index=False)

    print("\n" + "=" * 80)
    print("📊 4. PILAR CMP21 PATRÓN: ANÁLISIS ESTRATIFICADO POR Kt")
    print("=" * 80)
    df_te_pilar = df_te[~df_te["is_davis"]].copy()
    pilar_rows = []
    for b in kt_labels:
        sub = df_te_pilar[df_te_pilar["kt_bin"] == b]
        if len(sub) < 5:
            continue
        mA_res = calc_metrics_series(sub["target_GHI"].values, sub["pred_GHI_A"].values)
        mC_res = calc_metrics_series(sub["target_GHI"].values, sub["pred_GHI_C"].values)
        pilar_rows.append({
            "Rango_Kt": b,
            "N": len(sub),
            "GHI_Medio": mA_res["Mean_True"],
            "MBE_A": mA_res["MBE"],
            "MBE_C": mC_res["MBE"],
            "RMSE_A": mA_res["RMSE"],
            "RMSE_C": mC_res["RMSE"]
        })
    df_pilar_diag = pd.DataFrame(pilar_rows)
    print(df_pilar_diag.to_string(index=False))
    df_pilar_diag.to_csv(RESULTS_DIR / "fase4_dev_diagnostic_a_vs_c_pilar.csv", index=False)

    # 7. GENERACIÓN DE FIGURA DIAGNÓSTICA COMPLETA (6 PANELES)
    print("\n🎨 Generando figura diagnóstica de alta resolución...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=200)

    # Panel (a): MBE vs Kt
    x_kt = np.arange(len(df_kt_diag))
    width = 0.35
    axes[0, 0].bar(x_kt - width/2, df_kt_diag["MBE_A"], width, label="Opción A (t)", color="#1f77b4", edgecolor="black")
    axes[0, 0].bar(x_kt + width/2, df_kt_diag["MBE_C"], width, label="Opción C (mean t, t+1)", color="#2ca02c", edgecolor="black")
    axes[0, 0].axhline(0, color="black", linestyle="--", lw=1)
    axes[0, 0].set_xticks(x_kt)
    axes[0, 0].set_xticklabels([f"Kt {r.split()[0]}" for r in df_kt_diag["Rango_Kt"]], rotation=25, ha="right", fontsize=9)
    axes[0, 0].set_ylabel(r"Sesgo MBE ($\mathrm{W/m}^2$)", fontsize=11)
    axes[0, 0].set_title(r"(a) Sesgo (MBE) en función del Índice de Claridad $Kt$", fontsize=11, fontweight="bold")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[0, 0].legend(loc="lower left")

    # Panel (b): RMSE vs Kt
    axes[0, 1].bar(x_kt - width/2, df_kt_diag["RMSE_A"], width, label="Opción A (t)", color="#1f77b4", edgecolor="black")
    axes[0, 1].bar(x_kt + width/2, df_kt_diag["RMSE_C"], width, label="Opción C (mean t, t+1)", color="#2ca02c", edgecolor="black")
    axes[0, 1].set_xticks(x_kt)
    axes[0, 1].set_xticklabels([f"Kt {r.split()[0]}" for r in df_kt_diag["Rango_Kt"]], rotation=25, ha="right", fontsize=9)
    axes[0, 1].set_ylabel(r"RMSE GHI ($\mathrm{W/m}^2$)", fontsize=11)
    axes[0, 1].set_title(r"(b) Error Cuadrático Medio (RMSE) vs $Kt$", fontsize=11, fontweight="bold")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[0, 1].legend(loc="upper left")

    # Panel (c): MBE por Mes (Drift Temporal)
    x_mes = np.arange(len(df_mes_diag))
    axes[0, 2].plot(x_mes, df_mes_diag["MBE_A"], marker="o", color="#1f77b4", lw=2, label="Opción A (t)")
    axes[0, 2].plot(x_mes, df_mes_diag["MBE_C"], marker="s", color="#2ca02c", lw=2, label="Opción C (mean t, t+1)")
    axes[0, 2].axhline(0, color="black", linestyle="--", lw=1)
    axes[0, 2].set_xticks(x_mes)
    axes[0, 2].set_xticklabels(df_mes_diag["Mes"], rotation=30, ha="right", fontsize=9)
    axes[0, 2].set_ylabel(r"Sesgo MBE ($\mathrm{W/m}^2$)", fontsize=11)
    axes[0, 2].set_title(r"(c) Evolución Mensual del Sesgo (MBE 2021 OOS)", fontsize=11, fontweight="bold")
    axes[0, 2].grid(True, linestyle="--", alpha=0.5)
    axes[0, 2].legend(loc="lower left")

    # Panel (d): MBE vs cos Z
    x_cosz = np.arange(len(df_cosz_diag))
    axes[1, 0].bar(x_cosz - width/2, df_cosz_diag["MBE_A"], width, label="Opción A (t)", color="#1f77b4", edgecolor="black")
    axes[1, 0].bar(x_cosz + width/2, df_cosz_diag["MBE_C"], width, label="Opción C (mean t, t+1)", color="#2ca02c", edgecolor="black")
    axes[1, 0].axhline(0, color="black", linestyle="--", lw=1)
    axes[1, 0].set_xticks(x_cosz)
    axes[1, 0].set_xticklabels(["0.0-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"], fontsize=10)
    axes[1, 0].set_xlabel(r"$\cos Z$", fontsize=11)
    axes[1, 0].set_ylabel(r"Sesgo MBE ($\mathrm{W/m}^2$)", fontsize=11)
    axes[1, 0].set_title(r"(d) Sesgo (MBE) vs Elevación Solar ($\cos Z$)", fontsize=11, fontweight="bold")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[1, 0].legend(loc="lower left")

    # Panel (e): Pilar CMP21: RMSE vs Kt
    x_p = np.arange(len(df_pilar_diag))
    axes[1, 1].bar(x_p - width/2, df_pilar_diag["RMSE_A"], width, label="Opción A (t)", color="#1f77b4", edgecolor="black")
    axes[1, 1].bar(x_p + width/2, df_pilar_diag["RMSE_C"], width, label="Opción C (mean t, t+1)", color="#2ca02c", edgecolor="black")
    axes[1, 1].set_xticks(x_p)
    axes[1, 1].set_xticklabels([f"Kt {r.split()[0]}" for r in df_pilar_diag["Rango_Kt"]], rotation=25, ha="right", fontsize=9)
    axes[1, 1].set_ylabel(r"RMSE GHI ($\mathrm{W/m}^2$)", fontsize=11)
    axes[1, 1].set_title(r"(e) Pilar CMP21: RMSE en función de $Kt$", fontsize=11, fontweight="bold")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[1, 1].legend(loc="upper left")

    # Panel (f): Distribución de Residuales de Error
    err_A = df_te["error_A"].values
    err_C = df_te["error_C"].values
    axes[1, 2].hist(err_A, bins=80, range=(-600, 600), alpha=0.5, color="#1f77b4", density=True, label=f"Opción A (std={np.std(err_A):.1f})")
    axes[1, 2].hist(err_C, bins=80, range=(-600, 600), alpha=0.5, color="#2ca02c", density=True, label=f"Opción C (std={np.std(err_C):.1f})")
    axes[1, 2].axvline(0, color="black", linestyle="--", lw=1)
    axes[1, 2].set_xlabel(r"Residual $\hat{GHI} - GHI_{\mathrm{obs}}$ ($\mathrm{W/m}^2$)", fontsize=11)
    axes[1, 2].set_ylabel("Densidad de Probabilidad", fontsize=11)
    axes[1, 2].set_title(r"(f) Distribución de Residuales en Test 2021", fontsize=11, fontweight="bold")
    axes[1, 2].grid(True, linestyle="--", alpha=0.5)
    axes[1, 2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "figura_fase4_dev_diagnostic_a_vs_c.png")
    plt.savefig(ARTIFACT_DIR / "figura_fase4_dev_diagnostic_a_vs_c.png")
    plt.close()
    print("✓ Figura diagnóstica guardada en reportes/figures y artifact directory.")

    print("\n✅ DIAGNÓSTICO FINO COMPLETADO CON ÉXITO.")


if __name__ == "__main__":
    main()
