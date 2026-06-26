#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fase 3: Cálculo del Brillo Despejado (B0)
Ajusta la línea base de reflectancia en cielo claro para la provincia de Córdoba.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import json

INPUT_DIR = Path("data/ready/features_cell_hour")
OUTPUT_DIR = Path("data/ready/b0_model")

# Ecuación de Tarpley clear-sky (usando Fr en vez de B).
# B0 = A + B*cos(z) + C*sin(z)*cos(gamma) + D*sin(z)*cos^2(gamma)
def b0_model(X, A, B, C, D):
    cosZ, sinZ, cosGamma = X
    return A + B * cosZ + C * sinZ * cosGamma + D * sinZ * (cosGamma ** 2)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Leer todos los features
    files = sorted(list(INPUT_DIR.glob("features_*.parquet")))
    print(f"Cargando {len(files)} archivos de features...")
    
    df_list = []
    for f in files:
        df_list.append(pd.read_parquet(f))
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Radians
    df['zenith_rad'] = np.deg2rad(df['zenith_deg'])
    df['sinZ'] = np.sin(df['zenith_rad'])
    
    # Asumimos que la diferencia de azimut satélite-sol puede ser aproximada
    # por el azimuth solar directo como una variable periódica si asumimos sat constante.
    # Para Córdoba (aprox -64W, -32S) y GOES (75W), el acimut del satélite es estático y hacia el norte/noroeste.
    # Por simplicidad en esta fase, usaremos cos(azimuth_sun) que captura la dependencia geométrica.
    df['cosGamma'] = np.cos(np.deg2rad(df['azimuth']))
    
    # Filtrar datos problemáticos
    df = df[df['cosZ'] > 0.1].copy() # Solo día claro
    
    print("Determinando horas despejadas por celda...")
    # Selección: el cuantil de baja reflectancia per celda-hora para aislar cielo despejado.
    # Agrupamos por hora del día (0-23) y celda.
    df['hour'] = df['time_round'].dt.hour
    
    # Nos quedamos con el 15% inferior de reflectancia por celda y hora como proxy de "cielo despejado".
    def get_clear_sky(group):
        q = group['reflectance'].quantile(0.15)
        return group[group['reflectance'] <= q]
    
    df_clear = df.groupby(['cell_id', 'hour']).apply(get_clear_sky).reset_index(drop=True)
    
    print(f"Horas despejadas seleccionadas: {len(df_clear)}")
    
    # 2. Ajustar el modelo Global o por Región
    # Como no tenemos el DEM para regionalización topográfica, hacemos un único ajuste Global 
    # (o por celda, pero Global es más robusto si faltan datos en alguna celda).
    # Vamos a hacer un ajuste único global.
    
    X = (df_clear['cosZ'].values, df_clear['sinZ'].values, df_clear['cosGamma'].values)
    Y = df_clear['reflectance'].values
    
    print("Ajustando modelo BD-JPT / Tarpley B0...")
    # A=intercepto, B, C, D
    # Valores iniciales lógicos para reflectancia (esala de 0 a 1)
    popt, pcov = curve_fit(b0_model, X, Y, p0=[0.1, 0.1, 0.0, 0.0])
    
    model_params = {
        "A": popt[0],
        "B": popt[1],
        "C": popt[2],
        "D": popt[3],
        "MSE": float(np.mean((b0_model(X, *popt) - Y)**2))
    }
    
    print("Parámetros globales obtenidos para B0:")
    print(json.dumps(model_params, indent=2))
    
    with open(OUTPUT_DIR / "B0_params.json", "w") as f:
        json.dump(model_params, f, indent=2)
    
    # 3. Aplicar modelo B0 a todo el dataframe
    X_all = (df['cosZ'].values, df['sinZ'].values, df['cosGamma'].values)
    df['B0'] = b0_model(X_all, *popt)
    
    # Filtrar físicamente
    df['B0'] = df['B0'].clip(lower=0.01)
    
    print("Guardando features con B0...")
    df.to_parquet(OUTPUT_DIR / "features_with_B0.parquet", index=False)
    print("Fase 3 (Calculo de B0) finalizada.")

if __name__ == "__main__":
    main()
