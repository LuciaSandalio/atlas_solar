# 02 — Potencial Fotovoltaico Multianual (2020–2025)

Contiene los mapas ráster COG de rendimiento y producción fotovoltaica simulada a escala de utilidad (utility-scale) bajo estándares internacionales (IEC 61724, PVWatts v5 y modelo térmico NOCT), para configuración en plano fijo inclinado (30° Norte) y seguidor a un eje horizontal N-S con backtracking.

## Archivos y Especificaciones

| Archivo | Variable | Unidad | Rango Prov. Típico | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| `cordoba_poa_fijo_anual_medio_2020_2025_2km_cog.tif` | POA Fijo 30°N | kWh/m²/año | 1.042 – 2.206 | Irradiancia en plano fijo 30° N (Media Prov: 2.016,6 kWh/m²/año) |
| `cordoba_poa_tracker_anual_medio_2020_2025_2km_cog.tif` | POA Seguidor 1-eje | kWh/m²/año | 1.202 – 2.702 | Irradiancia con seguimiento solar 1-eje (Media Prov: 2.461,6 kWh/m²/año) |
| `cordoba_yf_fijo_anual_medio_2020_2025_2km_cog.tif` | Rendimiento Fijo ($Y_f$) | kWh/kWp/año | 871 – 1.793 | Producción específica sistema fijo (Media Prov: 1.644,7 kWh/kWp/año) |
| `cordoba_yf_tracker_anual_medio_2020_2025_2km_cog.tif` | Rendimiento Seguidor ($Y_f$) | kWh/kWp/año | 996 – 2.168 | Producción específica sistema seguidor (Media Prov: 1.982,2 kWh/kWp/año) |
| `cordoba_cf_fijo_anual_medio_2020_2025_2km_cog.tif` | Factor de Capacidad Fijo | % | 9,94% – 20,46% | Factor de planta anual fijo (Media Prov: 18,77%) |
| `cordoba_cf_tracker_anual_medio_2020_2025_2km_cog.tif` | Factor de Capacidad Seguidor | % | 11,37% – 24,74% | Factor de planta anual seguidor (Media Prov: 22,63%) |
| `cordoba_ganancia_tracker_anual_medio_2020_2025_2km_cog.tif` | Ganancia de Seguimiento | % | 14,38% – 23,36% | Incremento porcentual de generación del seguidor (Media Prov: +20,51%) |
