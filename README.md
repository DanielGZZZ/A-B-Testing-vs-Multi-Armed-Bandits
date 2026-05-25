# Optimización e Inferencia en Marketing Digital: A/B Testing frente a Multi-Armed Bandits

Este repositorio contiene el código fuente de las simulaciones/experimentación y de las gráficas de apoyo utilizadas en el Capitulo 2: Marco Teórico. El objetivo del proyecto es evaluar y comparar el rendimiento, la velocidad de aprendizaje y el coste de oportunidad entre las estrategias de experimentación tradicionales (A/B Testing) y los algoritmos adaptativos de Bandidos Multibrazo (Multi-Armed Bandits), con especial foco en el Muestreo de Thompson (Thompson Sampling).

## Estructura del Repositorio

El código está organizado en dos carpetas principales:

* `Experimentación/`: Contiene los scripts principales encargados de ejecutar las simulaciones de los usuarios interactuando con las variantes publicitarias.
    - `estacionario_alta_complejidad.py`: Simulación en un entorno estable con 6 anuncios/variantes.
    - `estacionario_baja_complejidad.py`: Simulación en un entorno estable con 3 anuncios/variantes.
    - `no_estacionario.py`: Simulación en un entorno dinámico donde las tasas de éxito de los anuncios cambian drásticamente a mitad de la campaña. Incorpora el mecanismo de **ventana deslizante de las últimas 1.000 interacciones** para evaluar la velocidad de reacción del algoritmo.
  
* `Visualizaciones_Marco_Teorico/`: Incluye los scripts en Python utilizados para generar las 5 ilustraciones estadísticas y conceptuales que dan soporte al bloque teórico de la memoria.
    - `graf_2_2.py`: Gráfica inspirada en el análisis de sesgo de exposición del blog de Netflix. La figura simula dos funciones de densidad de probabilidad gaussianas. Cada curva representa la creencia del algoritmo sobre el rendimiento de una opción. Las líneas verticales discontinuas indican el valor medio esperado (el punto de mayor probabilidad), mientras que la dispersión o anchura de cada campana refleja el grado de incertidumbre asociado a cada variante. (Sección 2.2 Dilema exploración vs explotación)
    - `graf_2_4_2.py`: Muestra la convergencia de la estimación Qt(a) frente al valor real q∗(a). Se observa cómo la alta volatilidad inicial (fasede exploración) se estabiliza conforme aumenta Nt(a), permitiendo una fase de explotación segura al reducirse el error de estimación. (Sección 2.4.2 Equilibrio dinámico entre Explorar y Explotar)
    - `graf_2_4_3.py`: Gráfica comparativa del arrepentimiento acumulado entre un test A/B y una estrategia MAB. (Sección 2.4.3 La métrica de optimización: El Arrepentimiento (Regret))
    - `graf1_2_4_5.py`: Representa la evolución temporal por impresiones de la distribución Beta para una sola acción con una tasa real del 25 % bajo aprendizaje bayesiano. (Sección 2.4.5 Estrategias de Selección de Acciones/ Muestreo de Thompson (Thompson Sampling))
    - `graf2_2_4_5.py`: Visualización de la actualización de la Distribución Beta en Thompson Sampling con varias variantes al mismo tiempo con su propia distribución independiente. (Sección 2.4.5 Estrategias de Selección de Acciones/ Muestreo de Thompson (Thompson Sampling))
