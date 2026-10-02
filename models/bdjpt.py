#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelos Físicos e Híbridos para el Atlas Solar de Córdoba (Convenio ATLAS II)
=============================================================================
Contiene las clases fundamentales:
- BDJPTWrapper: Modelo físico-empírico de transferencia radiativa (Tarpley / BD-JPT)
  adaptado al dominio del Índice de Claridad (Kt).
- HybridBDJPT_HGB: Modelo híbrido que combina la base física analítica BD-JPT con
  un ensamble de árboles gradient boosting (HistGradientBoostingRegressor) para
  la estimación no lineal de los residuales de nubosidad y atenuación atmosférica.
"""

import numpy as np
from scipy.optimize import lsq_linear
from sklearn.ensemble import HistGradientBoostingRegressor

class BDJPTWrapper:
    """
    Modelo físico-empírico BD-JPT (Suárez et al., 2012 / Tarpley, 1979).
    Ajusta coeficientes analíticos para cielo despejado (a, b, c) y atenuación
    nubosa (d_clear, d_cloudy) en función del contraste de reflectancia Fr.
    """
    def __init__(self, Isc: float = 1361.0):
        self.Isc = Isc
        self.a, self.b, self.c = 0.0, 0.0, 0.0
        self.d_clear, self.d_cloudy = 0.0, 0.0

    def fit(self, X: np.ndarray, target_ghi: np.ndarray):
        """
        X features: [reflectance, B0, Fr, cosZ, day_of_year]
        """
        ref, B0, Fr, cosZ = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        
        thresh_dynamic = B0 + 0.05
        mask_clear = ref <= thresh_dynamic
        mask_cloudy = ref > thresh_dynamic
        
        # Ajuste de cielo claro (a, b, c) con condiciones estrictamente despejadas
        mask_strictly_clear = ref <= (B0 + 0.02)
        if np.sum(mask_strictly_clear) < 500:
            mask_strictly_clear = ref <= np.quantile(ref, 0.15)
            
        X_strict = np.column_stack([
            self.Isc * cosZ[mask_strictly_clear],
            self.Isc * (cosZ[mask_strictly_clear]**2),
            self.Isc * (cosZ[mask_strictly_clear]**3)
        ])
        
        if len(X_strict) > 0:
            res_cs = lsq_linear(X_strict, target_ghi[mask_strictly_clear], bounds=(-3.0, 3.0))
            self.a, self.b, self.c = res_cs.x
        else:
            self.a, self.b, self.c = 1.0, 0.0, 0.0
            
        # Ajuste de índices de atenuación nubosa (d_clear, d_cloudy)
        GHI_cs_calc = self.Isc * cosZ * (self.a + self.b*cosZ + self.c*(cosZ**2))
        
        def fit_d(mask):
            if np.sum(mask) == 0:
                return 0.0
            X4 = Fr[mask].reshape(-1, 1)
            y_diff = target_ghi[mask] - GHI_cs_calc[mask]
            res = lsq_linear(X4, y_diff, bounds=(-10000.0, 0.0))
            return res.x[0]
            
        self.d_clear = fit_d(mask_clear)
        self.d_cloudy = fit_d(mask_cloudy)
        return self

    def predict(self, X: np.ndarray, ghi_cs_ineichen: np.ndarray) -> np.ndarray:
        """
        Retorna la predicción expresada en el dominio del Índice de Claridad (Kt).
        """
        ref, B0, Fr, cosZ = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        GHI_cs_calc = self.Isc * cosZ * (self.a + self.b*cosZ + self.c*(cosZ**2))
        
        pred_ghi = np.zeros(len(X), dtype=np.float64)
        
        thresh_dynamic = B0 + 0.05
        mask_clear = ref <= thresh_dynamic
        mask_cloudy = ref > thresh_dynamic
        
        pred_ghi[mask_clear] = GHI_cs_calc[mask_clear] + self.d_clear * Fr[mask_clear]
        pred_ghi[mask_cloudy] = GHI_cs_calc[mask_cloudy] + self.d_cloudy * Fr[mask_cloudy]
        
        pred_ghi = np.maximum(pred_ghi, 0.0)
        # Transformar a dominio Kt para compatibilidad con el pipeline
        denom = np.maximum(ghi_cs_ineichen, 1e-3)
        return pred_ghi / denom


class HybridBDJPT_HGB:
    """
    Arquitectura Híbrida: Estimador físico analítico BD-JPT como línea base
    + HistGradientBoostingRegressor para el aprendizaje supervisado de los residuales en Kt.
    """
    def __init__(self, lambda_shrinkage: float = 1.0, max_iter: int = 100, max_depth: int = 12, random_state: int = 42):
        self.bdjpt = BDJPTWrapper()
        self.hgb = HistGradientBoostingRegressor(max_iter=max_iter, max_depth=max_depth, random_state=random_state)
        self.lambda_shrinkage = lambda_shrinkage
        self.residual_mean = 0.0
        
    def fit(self, X: np.ndarray, y_kt: np.ndarray, y_ghi: np.ndarray, cs_tr: np.ndarray):
        self.bdjpt.fit(X, y_ghi)
        preds_kt_bdjpt = self.bdjpt.predict(X, cs_tr)
        
        residual = y_kt - preds_kt_bdjpt
        self.residual_mean = float(np.mean(residual))
        residual_centered = residual - self.residual_mean
        
        X_hybrid = np.column_stack([X, preds_kt_bdjpt])
        self.hgb.fit(X_hybrid, residual_centered)
        return self
        
    def predict(self, X: np.ndarray, cs_te: np.ndarray) -> np.ndarray:
        preds_kt_bdjpt = self.bdjpt.predict(X, cs_te)
        
        X_hybrid = np.column_stack([X, preds_kt_bdjpt])
        preds_residual = self.hgb.predict(X_hybrid) + self.residual_mean
        
        kt_hybrid = preds_kt_bdjpt + (self.lambda_shrinkage * preds_residual)
        return np.clip(kt_hybrid, 0.0, 1.15)
