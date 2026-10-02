import re

file_path = "/home/chula/.gemini/antigravity/brain/6c9b2b2f-9350-410d-8873-e5c094c49766/segundo_informe_tecnico.md"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 2. Inconsistencia 87.2% vs 20.5%
text = text.replace(
    r"la distribución de la tasa de retención diurna de la red es excepcionalmente alta y estable, con una **media provincial de retención del 87.2%**.",
    r"la tasa media de retención diurna efectiva general fue del **87.2%**. Sin embargo, la porción de datos estrictamente seleccionados bajo la ventana de cielo claro (esenciales para el cálculo de la calibración angular) representa en promedio un **20.5%** de los datos diurnos totales, lo cual constituye una muestra estadísticamente robusta."
)

text = text.replace(
    r"Distribución de la tasa de retención diurna.",
    r"Distribución de la fracción de datos seleccionados para calibración bajo cielo claro."
)

# 3. REST2 / Simplified Solis
text = text.replace(
    r"el **modelo REST2 / Simplified Solis**",
    r"un **modelo físico de cielo claro tipo Simplified Solis**, parametrizado con variables atmosféricas ERA5/CAMS"
)

# 4. Normas BSRN
text = text.replace(r"bajo normas BSRN", r"basado en criterios BSRN")
text = text.replace(r"basado en límites BSRN por Long y Dutton, 2002", r"basado en criterios BSRN (Long y Dutton, 2002)")

# 6. Cielos Despejados
text = text.replace(r"**Cielos Despejados**", r"**cielos despejados**")

# 7. Nulo
text = text.replace(r"convierte a Nulo", r"enmascarar como NaN")
text = text.replace(r"convertidos a Nulo", r"enmascarados como NaN")

# 8. Pilar referencia
text = text.replace(
    r"que funciona como la principal referencia observacional independiente para la calibración virtual.",
    r"que sirve como referencia radiométrica independiente para validar y anclar el sistema de calibración virtual."
)

# 9. RMSE y MBE
text = text.replace(
    r"el cambio en el Sesgo (MBE) demuestra la eliminación del error estructural del sensor. La sobrecorrección (MBE negativo) ocurre porque el sistema asume un cielo ideal REST2 libre de aerosoles anómalos.",
    r"La reducción del RMSE indica una mejora moderada de la consistencia general. La calibración reduce la sobreestimación extrema bajo cielo claro, aunque en la comparación agregada contra Pilar aparece un sesgo medio negativo. Este resultado sugiere una tendencia a la sobrecorrección en ciertos contextos, la cual puede estar asociada a limitaciones en la representación de aerosoles locales en CAMS, nubosidad fina o diferencias espaciales entre Pilar y Laguna Larga."
)

# 11. Regionalización text
text = text.replace(
    r"identificó **$k = 4$ clústeres independientes** como la partición más estable que explica la varianza provincial.",
    r"identificó **$k = 4$ clústeres independientes** como la partición más estable que explica la varianza provincial.\n\nLos clústeres obtenidos presentan una separación física coherente. El clúster 0 concentra estaciones con mayor GHI medio diurno e índice de claridad, asociado a condiciones atmosféricas más favorables o menor atenuación efectiva. El clúster 1 presenta menor GHI y menor Kt, compatible con zonas de mayor nubosidad relativa, mayor complejidad topográfica o sesgos instrumentales más marcados. El clúster 2 agrupa estaciones de baja altitud, principalmente hacia el este y sur provincial, con valores intermedios de GHI y claridad. El clúster 3 identifica enclaves de altura específicos en las sierras."
)

# 12. Fix the name from figura_villa_valeria_antes_despues to figura_13_serie_completa_villa_valeria
text = text.replace(
    r"figura_villa_valeria_antes_despues.png",
    r"figura_13_serie_completa_villa_valeria.png"
)

# Write back
with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Markdown replaced successfully")
