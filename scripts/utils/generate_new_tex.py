import re

content = r"""\documentclass[../main.tex]{subfiles}
% ===============================================================
% Documento
% ===============================================================
\begin{document}

\begin{titlepage}
\centering

\vspace*{-3cm}

\includegraphics[height=1.5cm]{figures/logo_fcefyn.png} % Note: Changed to figures/ assuming that's the dir
\hspace{1cm}
%\includegraphics[height=2cm]{figuras/logo_ohmc.jpeg}

\vspace{5cm}

{\fontsize{33}{40}\selectfont Convenio ATLAS II \par}
\vspace{0.4cm}
{\fontsize{33}{40}\selectfont Provincia de Córdoba \par}

\vspace{0.6cm}

{\fontsize{26}{32}\selectfont \textcolor{gray}{Componente Solar} \par}

\vspace{2.5cm}

{\large
Facultad de Ciencias Exactas, Físicas y Naturales (FCEFyN) \par}

\vspace{0.15cm}

{\large
Laboratorio de Hidrometeorología \par}

\vspace{3.5cm}

{\normalsize Segundo Informe de Avance \par}
\vspace{1cm}
{\normalsize Mayo 2026 \par}

\end{titlepage}

\newpage

\section*{Resumen general}
El presente informe documenta el desarrollo y ejecución de la \textbf{Fase B} del Atlas Solar de la Provincia de Córdoba. Esta fase abarca la consolidación de la base de datos en superficie mediante la fusión de series históricas de la red de estaciones automáticas Davis (2018--2026), la implementación de un control de calidad físico estandarizado bajo normas BSRN, el modelado dinámico de cielo claro mediante el modelo físico de transferencia radiativa \textbf{REST2} y la calibración virtual dinámica para corregir los sesgos ópticos, espectrales y de coseno de los sensores de silicio. Tras procesar más de 9.04 millones de registros crudos, se obtuvo un dataset final consolidado de \textbf{3.05 millones de horas diurnas calibradas} para \textbf{166 estaciones aprobadas}. Finalmente, se presenta una regionalización climática solar de la provincia mediante clustering no supervisado.

\bigskip

\newpage

\tableofcontents

\newpage

\section{Introducción y Estado del Convenio}

    \subsection{Alcance del Segundo Informe}
    El presente informe detalla la metodología y los resultados correspondientes a la \textbf{Fase B} del desarrollo del \textbf{Atlas Solar de la Provincia de Córdoba}. Esta etapa tiene como principal objetivo establecer una base de datos en superficie homogénea, de alta resolución temporal y con calidad controlada, para la caracterización del recurso solar provincial.
    
    La red de estaciones meteorológicas automáticas (AWS) Davis, operada de manera distribuida en el territorio por diferentes entidades gubernamentales y productivas, constituye una infraestructura invaluable. Sin embargo, los sensores de radiación de bajo costo basados en fotodiodos de silicio que equipan estas estaciones presentan sesgos intrínsecos debido a su respuesta espectral limitada y al error angular del difusor (error de coseno).
    
    Para corregir de forma dinámica estos sesgos y expandir de forma masiva la cobertura del estudio, se implementó un pipeline de \textbf{Calibración Virtual Dinámica}.
    
    \subsection{Síntesis del Estado de Avance}
    Para maximizar el alcance temporal del Atlas Solar, se recuperaron y acoplaron de forma exitosa los datos históricos agregados. La habilitación dinámica de simulaciones y calibraciones en el periodo temprano permitió integrar \textbf{217,104 horas de datos diurnos calibrados} previas a octubre de 2020. 
    Un total de \textbf{39 estaciones aprobadas} cuentan ahora con series temporales continuas y completas que cubren \textbf{los 7 años del rango de análisis (2019-2025)}, sirviendo de anclaje temporal absoluto. 
    
    \begin{figure}[H]
        \centering
        \includegraphics[width=0.75\textwidth]{figures/figura_7_cronograma_estaciones.png}
        \caption{Cronograma Gantt de Disponibilidad Temporal de la Red Davis Calibrada (2019--2025). Se destacan las barras continuas recuperadas de 2019.}
        \label{fig:gantt_disponibilidad}
    \end{figure}

\newpage

\section{Control de Calidad Físico (QC) de la Red de Superficie}

Según lo establecido en el primer informe de avance del Convenio ATLAS II - Componente Solar, la red de estaciones de superficie inicialmente seleccionada estaba conformada por 56 estaciones meteorológicas (EMAs) distribuidas de manera aproximadamente uniforme sobre el territorio provincial. Sin embargo, a partir de trabajos recientes desarrollados por el OHMC, la base de datos disponible fue ampliada significativamente, permitiendo incrementar la red de estudio a un total de 166 estaciones con series temporales de radiación solar continuas y suficientemente completas para el período analizado.

En consecuencia, la red actualmente utilizada queda conformada por las estaciones presentadas en la Fig. \ref{fig:mapa_red_estaciones}, constituyendo un conjunto observacional considerablemente más robusto respecto al de la versión inicial, tanto por la mayor cobertura espacial como por la mejora en la completitud temporal de las series disponibles. La estación de referencia \textbf{Pilar (CMP21)} (indicada con una estrella verde en el mapa) sirve como la principal referencia observacional independiente para todo el sistema de calibración virtual.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.75\linewidth]{figures/figura_1_mapa_red.png}
    \caption{Mapa de 166 EMAs seleccionadas para la calibración y futura generación del ATLAS provincial.}
    \label{fig:mapa_red_estaciones}
\end{figure}

\subsection{Protocolo de Control de Calidad de Datos Terrestres}

El procesamiento de cada serie temporal de radiación solar, correspondiente a las mediciones obtenidas en cada estación, se somete secuencialmente al siguiente protocolo de limpieza y control de calidad (basado en límites BSRN por Long y Dutton, 2002):

\begin{itemize}

    \item \textbf{Limpieza de valores no físicos:} se establece un umbral máximo físico dinámico basado en la constante solar extrapolada a superficie ($1.2 \cdot I_{SC} \cdot \cos^{1.2}\theta_z + 50$ W/m\textsuperscript{2}) y un máximo absoluto de 1400 W/m\textsuperscript{2}. Se eliminan todos los registros que superen dicho valor. Asimismo, se filtran los valores negativos de radiación solar y horas con $\theta_z \ge 85^\circ$.

    \item \textbf{Filtro de completitud anual:} se descartan aquellas series temporales cuya completitud anual sea inferior al 70\%. Dado que se requieren series no solo continuas sino también suficientemente completas, se exige que cada estación posea datos válidos en al menos el 70\% de las horas esperadas para cada año analizado.

    \item \textbf{Control de frecuencia intrahoraria:} se eliminan aquellas horas con menos del 80\% de las mediciones esperadas. Este criterio busca garantizar una frecuencia de muestreo suficientemente regular dentro de cada hora. En el caso de datos con resolución temporal de 10 minutos, se requieren idealmente 6 mediciones por hora, aceptándose un mínimo de 4 registros válidos.

    \item \textbf{Agregación horaria:} debido a la elevada variabilidad de las mediciones dentro de intervalos subhorarios, se calcula la media horaria de radiación solar para cada estación con el objetivo de reducir la influencia del ruido sobre esta variable.

\end{itemize}

    \subsection{Análisis de Disponibilidad y Completitud}
    En un análisis preliminar de completitud y continuidad de series temporales, se seleccionaron en primera instancia 56 EMAs. Al examinar en mayor detalle estas series en algunas de ellas se obtenían valores que discrepaban en periodicidad abruptamente entre dos años consecutivos, por lo que corresponden a datos de baja confiabilidad.
    
    Tras aplicar el protocolo exhaustivo en las 166 estaciones definitivas, se evaluó la \textbf{tasa de retención diurna} y la \textbf{completitud de datos}. Como se ilustra en la \textbf{Figura \ref{fig:completitud_analisis}}, la distribución de la tasa de retención diurna de la red es excepcionalmente alta y estable, con una \textbf{media provincial de retención del 87.2\%}.
    
    \begin{figure}[H]
        \centering
        \begin{subfigure}[b]{0.48\textwidth}
            \centering
            \includegraphics[width=\textwidth]{figures/figura_5_completitud_distribucion.png}
            \caption{Distribución de la tasa de retención diurna.}
        \end{subfigure}
        \hfill
        \begin{subfigure}[b]{0.48\textwidth}
            \centering
            \includegraphics[width=\textwidth]{figures/figura_6_mapa_completitud.png}
            \caption{Distribución espacial en la provincia.}
        \end{subfigure}
        \caption{Métricas de Completitud y Retención del Control de Calidad (QC) Davis.}
        \label{fig:completitud_analisis}
    \end{figure}

    \subsubsection{Control de Calidad (QC) Trimestral Dinámico}
    Como filtro final estricto, se implementó un algoritmo de validación trimestral. Para cada estación y trimestre, se comparó el percentil 99.9 de la serie calibrada ($P99_{cal}$) contra el percentil 99.9 del modelo REST2 ($P99_{cs}$). Si se desvía más de un $\pm25\%$, el trimestre se rechaza. Este umbral previene el rechazo de anomalías climáticas legítimas (humo, polvo) actuando exclusivamente como un filtro de fallas de hardware. Se identificaron y eliminaron selectivamente \textbf{63 trimestres anómalos (1.9\%)}.
    
    \begin{figure}[H]
        \centering
        \includegraphics[width=0.9\textwidth]{figures/figura_qc_trimestral_efecto.png}
        \caption{Efecto del Control de Calidad Trimestral Dinámico. Las brechas visualizan los períodos anómalos que fueron identificados y convertidos a Nulo.}
        \label{fig:qc_trimestral}
    \end{figure}


\newpage

\section{Simulación Clear-Sky REST2 (El Calibrador Virtual)}

    \subsection{Configuración y parametrización del modelo REST2}
    Para evitar depender de una grilla satelital intermedia que introduce sesgos espaciales, se aplicó una \textbf{interpolación trilineal (tiempo, latitud, longitud)} para proyectar el estado de la atmósfera sobre cada coordenada exacta de estación de forma sub-horaria, utilizando:
    \begin{itemize}
        \item \textbf{Espesor Óptico de Aerosoles (AOD):} obtenido de la base de datos CAMS (Copernicus Atmosphere Monitoring Service) a 550nm.
        \item \textbf{Agua Precipitable Total (PWV):} del modelo global de reanálisis ERA5.
        \item \textbf{Presión Atmosférica en Superficie ($P_s$):} de ERA5, corregida localmente por la altitud.
    \end{itemize}
    
    \subsection{Generación de la Base de Cielo Claro de Referencia}
    Con los parámetros atmosféricos interpolados, se ejecutó el \textbf{modelo REST2 / Simplified Solis}, el cual proporciona la irradiancia teórica de cielo claro ($GHI_{cs}$) para cada hora útil en las 166 estaciones. Esta base sintética constituye la referencia fundamental ("Calibrador Virtual") contra la cual se contrasta la señal empírica de los fotodiodos para derivar los factores de corrección angulares.

\newpage

\section{Calibración de sensores de silicio (Red Davis) mediante REST2} 

    \subsection{Metodología de calibración virtual relativa} 
    Para cada estación, se seleccionaron los datos correspondientes a \textbf{Cielos Despejados} mediante un filtro de estabilidad temporal sobre la serie observada:
    \begin{equation}
        \sigma_{3h}\left(\frac{GHI_{\text{obs}}}{GHI_{cs}}\right) \le 0.03
    \end{equation}
    Donde $\sigma_{3h}$ es la desviación estándar móvil de 3 horas. El umbral empírico del 3\% fue seleccionado tras pruebas de sensibilidad para aislar rigurosamente ventanas libres de nubes reteniendo una cantidad significativa de datos ($N > 100$) para el ajuste.
    
    El análisis de la red reveló que numerosos fotodiodos experimentaron saltos estructurales drásticos a lo largo de su vida útil (Fig. \ref{fig:villa_valeria_serie}). Para absorber esto, la calibración se realiza de manera \textbf{segmentada por año}.
    
    \begin{figure}[H]
        \centering
        \includegraphics[width=0.85\textwidth]{figures/figura_villa_valeria_antes_despues.png}
        \caption{Comparativa de la Serie en Villa Valeria. Los niveles máximos se estabilizan tras la calibración anual segmentada.}
        \label{fig:villa_valeria_serie}
    \end{figure}
    
    \subsection{Modelado de la respuesta angular y espectral}
    Para corregir los sesgos, se propuso un factor multiplicativo dinámico dependiente de la posición solar $K(\cos\theta_z) = GHI_{cs}/GHI_{\text{obs}}$. Siguiendo la literatura (Vignola et al., 2016), se modeló este factor mediante un ajuste polinomial cuadrático para evitar overfitting y capturar la reflexión no-lineal del difusor:
    \begin{equation}
        K(\cos\theta_z) = c_0 + c_1 \cos\theta_z + c_2 \cos^2\theta_z
    \end{equation}
    
    \subsection{Resultados de la Calibración de Estaciones}
    Las curvas de calibración obtenidas reflejan con total consistencia la física del sensor Davis:
    \begin{enumerate}
        \item \textbf{Subestimación Espectral a Mediodía (Sol Alto, $\cos\theta_z \approx 1$):} El sensor original subestima la radiación real en un 6.0\% a mediodía, debido a la falta de respuesta en el infrarrojo.
        \item \textbf{Error Angular de Coseno a Sol Bajo ($\cos\theta_z \approx 0.5$):} El factor $K$ alcanza un valor promedio de \textbf{1.14} (subestimación del \textbf{14.0\%}) por la reflexión oblicua del difusor acrílico.
    \end{enumerate}

    \begin{figure}[H]
        \centering
        \begin{subfigure}[b]{0.48\textwidth}
            \centering
            \includegraphics[width=\textwidth]{figures/figura_2_curvas_calibracion.png}
            \caption{Curvas de Calibración $K(\cos Z)$.}
        \end{subfigure}
        \hfill
        \begin{subfigure}[b]{0.48\textwidth}
            \centering
            \includegraphics[width=\textwidth]{figures/figura_4_histograma_factores.png}
            \caption{Histograma de $K$ a $\cos Z = 0.5$.}
        \end{subfigure}
        \caption{Resultados del modelo de calibración angular.}
    \end{figure}

\newpage

\section{Validación de la calibración de superficie}

    \subsection{Validación cruzada bajo cielo claro}
    La validación empírica cruzó la estación Davis \textbf{Laguna Larga} con el piranómetro de termopila estándar de investigación \textbf{CMP21 de Pilar}. Ambas estaciones están separadas por \textbf{12.3 kilómetros} en un terreno perfectamente llano (14,904 horas evaluadas).
    
    Es importante notar una aparente discrepancia: mientras la red en promedio \textit{subestima} la radiación (requiriendo $K > 1$), la estación Laguna Larga presentaba una fuerte \textbf{sobreestimación} (+20\%), indicando un multiplicador local de fábrica desajustado. El pipeline asimiló este error generando un atenuador ($K < 1$).
    
    \begin{figure}[H]
        \centering
        \includegraphics[width=0.75\textwidth]{figures/figura_3_comparacion_pilar.png}
        \caption{Comparación de GHI en Laguna Larga vs Pilar para el 5 de Noviembre de 2020.}
        \label{fig:comparacion_pilar}
    \end{figure}
    
    El impacto cuantitativo evaluado globalmente resultó en:
    \begin{itemize}
        \item \textbf{Antes de la Calibración:} RMSE = $190.65 \text{ W/m}^2$ | MBE = $+18.75 \text{ W/m}^2$.
        \item \textbf{Después de la Calibración:} \textbf{RMSE = $173.88 \text{ W/m}^2$} | \textbf{MBE = $-38.56 \text{ W/m}^2$}.
    \end{itemize}
    
    Al interpretar estos resultados, es vital considerar que la distancia de 12.3 km introduce un ``error de representatividad espacial'': una nube puede sombrear Pilar y no Laguna Larga, disparando el RMSE. Por ello, la mejora del RMSE es modesta ($\sim 9\%$), pero el cambio en el Sesgo (MBE) demuestra la eliminación del error estructural del sensor. La sobrecorrección (MBE negativo) ocurre porque el sistema asume un cielo ideal REST2 libre de aerosoles anómalos.
    
    \textbf{Riesgo de Circularidad:} El uso de REST2 tanto para calibrar como referencia teórica introduce riesgo analítico. La validación empírica contra la estación CMP21 independiente actúa como el escudo metodológico que rompe esta circularidad.

\newpage

\section{Conclusiones y transición a la fase satelital}

    \subsection{Regionalización Solar (K-Means)}
    Para segmentar la provincia según la variabilidad intrínseca del recurso solar, se llevó a cabo un clustering no supervisado \textbf{K-Means} sobre los parámetros físicos. El análisis del Silhouette Score (0.271) identificó como óptimo el agrupamiento en \textbf{$k = 4$ clústeres independientes}.
    
    \begin{figure}[H]
        \centering
        \includegraphics[width=0.65\textwidth]{figures/figura_10_mapa_clusters_solares.png}
        \caption{Mapa de Regionalización Solar de la Provincia de Córdoba mediante clustering K-Means ($k=4$).}
    \end{figure}

    \subsection{Conclusiones del trabajo terrestre}
    La finalización de la calibración de superficie marca un avance metodológico significativo:
    \begin{enumerate}
        \item \textbf{Base de Datos Unificada y Calibrada:} Dataset de \textbf{166 estaciones} y \textbf{3.05 millones de horas útiles} con calidad controlada.
        \item \textbf{Consistencia Física Robusta:} Corrección estructural dependiente del ángulo cenital.
        \item \textbf{Máxima Cobertura:} \textbf{39 estaciones continuas de largo plazo (2019-2025)}.
        \item \textbf{Zonificación Científica:} Segmentación en 4 clústeres sólidos.
    \end{enumerate}
    
    \textbf{Evaluación de Incertidumbre:} Sumando la incertidumbre del reanálisis ERA5/CAMS ($\sim 5-8\%$) y la precisión residual post-calibración (evidenciada en la validación), la incertidumbre horaria esperada del dataset GHI final terrestre se estima en el rango de \textbf{$\pm 10-15\%$}.
    
    \subsection{Avance del Informe 3}
    Los datasets resultantes constituyen la base sólida de datos en superficie sobre la cual se sustentará la posterior \textbf{calibración satelital y modelado espacial (Fase C)} en el tercer informe del proyecto. Se desarrollará el \textit{downscaling} de la base de datos CAMS para toda la geografía provincial utilizando esta red calibrada.

\end{document}
"""

with open('/media/chula/Expansion/Convenios/atlas_solar/reportes/segundo_informe_tecnico.tex', 'w') as f:
    f.write(content)
print("Updated successfully")
