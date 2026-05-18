import numpy as np
import matplotlib.pyplot as plt

class EntornoPublicitario:
    def __init__(self):
        """Simula un entorno publicitario con tres anuncios diferentes y sus tasas de clics reales (ocultas para el algoritmo).
        
        Las tasas de clics son:
        - Anuncio_A: 4% (Huevo blanco, neutro)
        - Anuncio_B: 5% (Mayonesa amarilla, ideal)
        - Anuncio_C: 2% (Mayonesa blanca, error cultural)
        """
        
        self.tasas_reales = {
            'Anuncio_A': 0.04, # Neutro
            'Anuncio_B': 0.05, # Ideal
            'Anuncio_C': 0.02  # Error cultural
        }
        self.acciones = list(self.tasas_reales.keys())

    def generar_respuesta_usuario(self, accion):
        """Simula la respuesta del usuario a un anuncio haciendo clic o no usando una distribución de Bernoulli.
        Args:            accion (str): El anuncio mostrado al usuario.
        Returns:            bool: True si el usuario hizo clic, False en caso contrario.
        """
        probabilidad = self.tasas_reales[accion]
        return np.random.rand() < probabilidad

# Inicialización de Priors (Beta(1,1))
alphas = {}
for a in ['Anuncio_A', 'Anuncio_B', 'Anuncio_C']:
    alphas[a] = 1
    
betas = {}
for a in ['Anuncio_A', 'Anuncio_B', 'Anuncio_C']:
    betas[a] = 1

# THOMPSON SAMPLING (Exploración adaptativa)
def thompson_sampling_step(alphas, betas):
    """Realiza un paso de Thompson Sampling para elegir un anuncio.
    Args:        
        alphas (dict): Parámetros alpha de las distribuciones Beta para cada anuncio.
        betas (dict): Parámetros beta de las distribuciones Beta para cada anuncio. 
    Returns:        str: El anuncio elegido para mostrar al usuario.
    """
    muestras = {} # Almacena lo que el algoritmo cree que es la tasa de clics de cada anuncio en este momento.
    for a in alphas.keys():
        # .beta() genera una muestra aleatoria de la distribución Beta definida por los parámetros alpha y beta actuales para cada anuncio.
        muestras[a] = np.random.beta(alphas[a], betas[a])
    
    accion_elegida = max(muestras, key=muestras.get) # Elige el anuncio con la muestra más alta, es decir, el que el algoritmo considera más prometedor en este momento.
    return accion_elegida


# A/B TESTING (Exploración fija al inicio)
anuncio_ganador = None # Variable para almacenar el anuncio ganador después del periodo de exploración
def ab_testing_step(t, alphas, betas, periodo_exploracion=1500): 
    """Realiza un paso de A/B Testing con un periodo de exploración fijo.
    Args:        
        t (int): El número de intento actual.
        alphas (dict): Parámetros alpha de las distribuciones Beta para cada anuncio.
        betas (dict): Parámetros beta de las distribuciones Beta para cada anuncio.
        periodo_exploracion (int): Número de intentos dedicados a la exploración antes de explotar el mejor anuncio. En este caso, 2500 intentos (12.5% de 20.000).
    Returns:        str: El anuncio elegido para mostrar al usuario.
    """
    global anuncio_ganador
    if t < periodo_exploracion: # t es cada intento.
        # Exploración equitativa: rotar entre los anuncios
        acciones = ['Anuncio_A', 'Anuncio_B', 'Anuncio_C']
        indice = t % 3  # Esto da 0, 1 o 2
        return acciones[indice] # Durante los primeros 1500 intentos, se muestra cada anuncio de manera equitativa
    
    else:        # Explotación: elegir el anuncio con mejor tasa observada. A partir del 1501 intento.
        if anuncio_ganador is None:
            mejor_tasa = -1 # Inicializamos con un valor muy bajo para asegurar que cualquier tasa calculada sea mejor
            for a in ['Anuncio_A', 'Anuncio_B', 'Anuncio_C']:
                intentos_totales = alphas[a] + betas[a]
                tasa = alphas[a] / intentos_totales 
    
                if tasa > mejor_tasa: 
                    mejor_tasa = tasa
                    anuncio_ganador = a
            
        return anuncio_ganador


# EPSILON-GREEDY (Exploración constante) 
def epsilon_greedy_step(alphas, betas, epsilon=0.1):
    """Realiza un paso de Epsilon-Greedy con una tasa de exploración constante.
    Args:        
        alphas (dict): Parámetros alpha de las distribuciones Beta para cada anuncio.
        betas (dict): Parámetros beta de las distribuciones Beta para cada anuncio.
        epsilon (float): La probabilidad de elegir un anuncio aleatorio en lugar del mejor. En este caso, 0.1 (10% de exploración).
    Returns:        str: El anuncio elegido para mostrar al usuario.
    """
    # random.rand() genera un número aleatorio entre 0 y 1. Si este número es menor que epsilon, se elige un anuncio aleatorio (exploración). 
    if np.random.rand() < epsilon: # Si el número aleatorio es menor que epsilon (0.1), se realiza una exploración. (10% de las veces)
        # Exploración aleatoria
        return np.random.choice(list(alphas.keys())) # choice() selecciona aleatoriamente un anuncio de la lista de anuncios disponibles.
    else:
        # De lo contrario, se elige el anuncio con la mejor tasa observada (explotación).
        # Para cada anuncio, se calcula la tasa de clics observada como alphas[a] / (alphas[a] + betas[a]), que representa la proporción de éxitos (clics) sobre el total de intentos para ese anuncio.
        # Luego, se selecciona el anuncio con la tasa más alta utilizando max() con una función lambda que compara las tasas de clics de cada anuncio.
        return max(alphas, key=lambda a: alphas[a] / (alphas[a] + betas[a]))

entorno = EntornoPublicitario()
intentos = 50000 
num_experimentos = 100
estrategias = ['AB_Testing', 'Epsilon_Greedy', 'Thompson_Sampling']

resultados_globales = {est: [] for est in estrategias}

for est in estrategias:
    for i in range(num_experimentos):
        entorno = EntornoPublicitario()
        alphas = {a: 1 for a in entorno.acciones}
        betas = {a: 1 for a in entorno.acciones}
        regret_total = 0
        regret_acumulado_instante = []
        
        for t in range(intentos):
            if est == 'AB_Testing':
                a_t = ab_testing_step(t, alphas, betas)
            elif est == 'Epsilon_Greedy':
                a_t = epsilon_greedy_step(alphas, betas)
            else:
                a_t = thompson_sampling_step(alphas, betas)
                
            exito = entorno.generar_respuesta_usuario(a_t)
            regret_total += (0.05 - entorno.tasas_reales[a_t])
            regret_acumulado_instante.append(regret_total)
            
            if exito: 
                alphas[a_t] += 1
            else: 
                betas[a_t] += 1
        
        resultados_globales[est].append(regret_acumulado_instante)

# Cálculo de la media y desviación estándar para cada estrategia. Grafica con áreas de varianza y muestra el regret medio final al terminar la simulación (+/- 1 desviación estándar).
plt.figure(figsize=(12, 7))
colores = {'AB_Testing': 'blue', 'Epsilon_Greedy': 'orange', 'Thompson_Sampling': 'green'}

for est in estrategias:
    datos = np.array(resultados_globales[est])
    media = np.mean(datos, axis=0)
    desviacion = np.std(datos, axis=0)
    
    plt.plot(media, label=est, color=colores[est], linewidth=2.5)
    plt.fill_between(range(intentos), media - desviacion, media + desviacion, color=colores[est], alpha=0.15)
    print(f"{est}: Regret medio final = {media[-1]:.4f}")

plt.xlabel('Número de Usuarios (Tiempo t)', fontsize=14, fontweight='bold')
plt.ylabel('Regret Acumulado Medio', fontsize=14, fontweight='bold')
plt.title(f'Comparativa Estacionaria: 3 Anuncios (Promedio de {num_experimentos} {"simulación" if num_experimentos == 1 else "simulaciones"})', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tick_params(labelsize=12)
plt.show()
