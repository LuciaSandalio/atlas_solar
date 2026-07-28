import re

file_path = "/media/chula/Expansion/Convenios/atlas_solar/reportes/segundo_informe_tecnico.tex"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Resumen General
text = re.sub(
    r"El presente informe documenta el desarrollo y ejecución.*?(?=\n\n\s*\\bigskip)",
    "El presente informe documenta el desarrollo y ejecución de la \\textbf{Fase B} del Atlas Solar de la Provincia de Córdoba, correspondiente a la consolidación y calibración de la red terrestre de radiación solar. Esta etapa abarca la integración de series históricas de estaciones automáticas Davis, la aplicación de un control de calidad físico basado en criterios BSRN, la simulación de irradiancia de cielo claro mediante un modelo físico parametrizado con ERA5/CAMS y la implementación de una calibración virtual dinámica para corregir sesgos angulares, espectrales y de escala en sensores de silicio. Tras el procesamiento de más de 9.04 millones de registros crudos, se obtuvo un conjunto consolidado de 3.05 millones de horas diurnas calibradas correspondientes a 166 estaciones aprobadas. Finalmente, se presenta una regionalización solar preliminar de la provincia mediante clustering no supervisado.",
    text, flags=re.DOTALL
)

# 2. Inconsistencia 87.2% vs 20.5%
text = text.replace(
    r"la distribución de la tasa de retención diurna de la red es excepcionalmente alta y estable, con una \textbf{media provincial de retención del 87.2\%}.",
    r"la tasa media de retención diurna efectiva general fue del \textbf{87.2\%}. Sin embargo, la porción de datos estrictamente seleccionados bajo la ventana de cielo claro (esenciales para el cálculo de la calibración angular) representa en promedio un \textbf{20.5\%} de los datos diurnos totales, lo cual constituye una muestra estadísticamente robusta."
)
text = text.replace(
    r"\caption{Distribución de la tasa de retención diurna.}",
    r"\caption{Distribución de la fracción de datos seleccionados para calibración bajo cielo claro.}"
)

# 3. REST2 / Simplified Solis
text = text.replace(
    r"el \textbf{modelo REST2 / Simplified Solis}",
    r"un \textbf{modelo físico de cielo claro tipo Simplified Solis}, parametrizado con variables atmosféricas ERA5/CAMS"
)

# 4. Normas BSRN
text = text.replace(
    r"bajo normas BSRN",
    r"basado en criterios BSRN"
)
text = text.replace(
    r"basado en límites BSRN por Long y Dutton, 2002",
    r"basado en criterios BSRN (Long y Dutton, 2002)"
)

# 5. Markdown en LaTeX
text = text.replace(r"**Villa Valeria**", r"\textbf{Villa Valeria}")
text = text.replace(r"**fuerte sobreestimación**", r"\textbf{fuerte sobreestimación}")

# 6. Cielos Despejados -> cielos despejados
text = text.replace(r"\textbf{Cielos Despejados}", r"\textbf{cielos despejados}")

# 7. Nulo -> valores faltantes
text = text.replace(r"convertirse a Nulo", r"enmascarados como NaN")
text = text.replace(r"convierte a Nulo", r"enmascarar como NaN")
text = text.replace(r"convertidos a Nulo", r"enmascarados como NaN")

# 8. Pilar referencia
text = text.replace(
    r"sirve como la principal referencia observacional independiente para todo el sistema de calibración virtual.",
    r"sirve como referencia radiométrica independiente para validar y anclar el sistema de calibración virtual."
)

# 9. RMSE y MBE Validación
text = text.replace(
    r"el cambio en el Sesgo (MBE) demuestra la eliminación del error estructural del sensor. La sobrecorrección (MBE negativo) ocurre porque el sistema asume un cielo ideal REST2 libre de aerosoles anómalos.",
    r"La reducción del RMSE indica una mejora moderada de la consistencia general. La calibración reduce la sobreestimación extrema bajo cielo claro, aunque en la comparación agregada contra Pilar aparece un sesgo medio negativo. Este resultado sugiere una tendencia a la sobrecorrección en ciertos contextos, la cual puede estar asociada a limitaciones en la representación de aerosoles locales en CAMS, nubosidad fina o diferencias espaciales entre Pilar y Laguna Larga."
)

# 10. Labels en figuras
text = text.replace(
    r"\caption{Resultados del modelo de calibración angular.}",
    r"\caption{Resultados del modelo de calibración angular.}\n        \label{fig:curvas_calibracion}"
)
text = text.replace(
    r"\caption{Mapa de Regionalización Solar de la Provincia de Córdoba mediante clustering K-Means ($k=4$).}",
    r"\caption{Mapa de Regionalización Solar de la Provincia de Córdoba mediante clustering K-Means ($k=4$).}\n        \label{fig:mapa_clusters}"
)

# 11. Regionalización: Section instead of Subsection
text = text.replace(
    r"\subsection{Regionalización Solar (K-Means)}",
    r"\section{Regionalización solar y variabilidad espacial de la calibración}\n\n    \subsection{Zonificación mediante Clustering Multivariado}"
)
text = text.replace(
    r"identificó como óptimo el agrupamiento en \textbf{$k = 4$ clústeres independientes}.",
    r"identificó como óptimo el agrupamiento en \textbf{$k = 4$ clústeres independientes}.\n\n    Los clústeres obtenidos presentan una separación física coherente. El clúster 0 concentra estaciones con mayor GHI medio diurno e índice de claridad, asociado a condiciones atmosféricas más favorables o menor atenuación efectiva. El clúster 1 presenta menor GHI y menor Kt, compatible con zonas de mayor nubosidad relativa, mayor complejidad topográfica o sesgos instrumentales más marcados. El clúster 2 agrupa estaciones de baja altitud, principalmente hacia el este y sur provincial, con valores intermedios de GHI y claridad. El clúster 3 identifica enclaves de altura específicos en las sierras."
)
text = text.replace(
    r"Segmentación en 4 clústeres sólidos.",
    r"Segmentación en 4 regiones solares principales."
)

# Insert the new figures into Regionalization section
text = text.replace(
    r"\subsection{Conclusiones del trabajo terrestre}",
    r"""    \subsection{Variabilidad regional de la calibración}
    El análisis de los factores de calibración a nivel regional evidencia cómo las distintas condiciones geográficas imponen diferentes exigencias ópticas a los sensores Davis. La \textbf{Figura \ref{fig:curvas_regionales}} muestra los ajustes polinomiales medios para cada uno de los clústeres identificados, mientras que la \textbf{Figura \ref{fig:comparacion_regional}} contrasta las series temporales de estaciones representativas de cada región frente al modelo de cielo claro.
    
    \begin{figure}[H]
        \centering
        \includegraphics[width=0.75\textwidth]{figures/figura_8_curvas_regionales.png}
        \caption{Curvas de calibración polinomial segmentadas por clúster solar.}
        \label{fig:curvas_regionales}
    \end{figure}

    \begin{figure}[H]
        \centering
        \includegraphics[width=0.9\textwidth]{figures/figura_9_comparacion_regional.png}
        \caption{Comparación regional de perfiles horarios en día despejado. Se aprecia la variabilidad de la corrección según el emplazamiento de la estación.}
        \label{fig:comparacion_regional}
    \end{figure}
    
    \subsection{Conclusiones del trabajo terrestre}"""
)

# 12. Fix the name from figura_villa_valeria_antes_despues to figura_13_serie_completa_villa_valeria
text = text.replace(
    r"figura_villa_valeria_antes_despues.png",
    r"figura_13_serie_completa_villa_valeria.png"
)

# Write back
with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Text replaced successfully")
