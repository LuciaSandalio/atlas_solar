# 05 — Variabilidad Interanual y Convergencia Temporal

Contiene los mapas ráster COG que cuantifican la dispersión interanual del recurso solar y la estabilización asintótica de la media climatológica multianual.

## Archivos y Métricas

- `cordoba_cv_interanual_2020_2025_2km_cog.tif`: Coeficiente de variación interanual sexenal ($\sigma / \mu \times 100\%$). Cuantifica el riesgo de incertidumbre interanual. Media provincial: **2,14%** (rango: 0,47% en llanura a 16,99% en cumbres).
- `cordoba_delta_2_pct_2km_cog.tif` a `cordoba_delta_6_pct_2km_cog.tif`: Sensibilidad incremental $\Delta_k = 100 \times (\bar{H}_k - \bar{H}_{k-1}) / \bar{H}_{k-1}$. Demuestra cómo las oscilaciones se amortiguan (|$\Delta_6$| media provincial = **0,15%**), garantizando representatividad climática.
