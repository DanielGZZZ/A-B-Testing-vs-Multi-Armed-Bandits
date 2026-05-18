import numpy as np
import matplotlib.pyplot as plt

class EntornoPublicitario:
    def __init__(self):
        # Escenario de 6 anuncios: Variantes similares para aumentar dificultad
        self.tasas_reales = {
            'Anuncio_A1': 0.04,  # Neutro 1
            'Anuncio_A2': 0.038, # Neutro 2 (Variante)
            'Anuncio_B1': 0.05,  # Ideal 1 (Ganador)
            'Anuncio_B2': 0.042, # Ideal 2 (Variante muy cercana)
            'Anuncio_C1': 0.02,  # Error cultural 1
            'Anuncio_C2': 0.018  # Error cultural 2 (Variante)
        }
        self.acciones = list(self.tasas_reales.keys())
        self.mejor_tasa = max(self.tasas_reales.values())

    def generar_respuesta_usuario(self, accion):
        """Simula la respuesta del usuario a un anuncio haciendo clic o no usando una distribución de Bernoulli.
        Args:            accion (str): El anuncio mostrado al usuario.
        Returns:            bool: True si el usuario hizo clic, False en caso contrario."""
        probabilidad = self.tasas_reales[accion]
        return np.random.rand() < probabilidad

# THOMPSON SAMPLING (Exploración adaptativa) 
def thompson_sampling_step(alphas, betas):
    """Realiza un paso de Thompson Sampling para elegir un anuncio.
    Args:        
        alphas (dict): Parámetros alpha de las distribuciones Beta para cada anuncio.
        betas (dict): Parámetros beta de las distribuciones Beta para cada anuncio.
    Returns:        str: El anuncio elegido para mostrar al usuario.
    """
    muestras = {}
    for a in alphas.keys():
        muestras[a] = np.random.beta(alphas[a], betas[a])
    return max(muestras, key=muestras.get)


# A/B TESTING (Exploración fija al inicio)
ganador_ab = None # Variable global para guardar la decisión

def ab_testing_step(t, alphas, betas, periodo_exploracion=1500):
    """Realiza un paso de A/B Testing con un periodo de exploración fijo.
    Args:        
        t (int): El número de intento actual.
        alphas (dict): Parámetros alpha de las distribuciones Beta para cada anuncio.
        betas (dict): Parámetros beta de las distribuciones Beta para cada anuncio.
        periodo_exploracion (int): Número de intentos dedicados a la exploración pura. 
    Returns:        str: El anuncio elegido para mostrar al usuario.
    """
    global ganador_ab
    acciones = list(alphas.keys())
    
    # Exploración pura
    if t < periodo_exploracion:
        indice = t % len(acciones)
        return acciones[indice]
    
    # Decisión (Solo se ejecuta una vez en el usuario 1500)
    elif t == periodo_exploracion:
        mejor_tasa = -1
        for a in acciones:
            tasa = alphas[a] / (alphas[a] + betas[a])
            if tasa > mejor_tasa:
                mejor_tasa = tasa
                ganador_ab = a
        return ganador_ab
    
    # Explotación fija (Ya no mira los datos nuevos solo aplica el ganador)
    else:
        return ganador_ab

# EPSILON-GREEDY (Exploración aleatoria con probabilidad fija)
def epsilon_greedy_step(alphas, betas, epsilon=0.1):
    """Realiza un paso de Epsilon-Greedy para elegir un anuncio.
    Args:        
        alphas (dict): Parámetros alpha de las distribuciones Beta para cada anuncio.
        betas (dict): Parámetros beta de las distribuciones Beta para cada anuncio.
        epsilon (float): Probabilidad de explorar (elegir un anuncio aleatorio).
    Returns:        str: El anuncio elegido para mostrar al usuario.
    """
    if np.random.rand() < epsilon:
        return np.random.choice(list(alphas.keys()))
    else:
        return max(alphas, key=lambda a: alphas[a] / (alphas[a] + betas[a]))

# Simulacion para 6 anuncios con las 3 estrategias
intentos = 50000
num_experimentos = 100
estrategias = ['AB_Testing', 'Epsilon_Greedy', 'Thompson_Sampling']
resultados = {est: [] for est in estrategias}

for est in estrategias:
    for i in range(num_experimentos):
        ganador_ab = None  # Reiniciar el ganador para cada experimento
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
            
            # El mejor CTR sigue siendo 0.05 (Anuncio_B1)
            regret_total += (0.05 - entorno.tasas_reales[a_t])
            regret_acumulado_instante.append(regret_total)
            
            if exito: 
                alphas[a_t] += 1
            else: 
                betas[a_t] += 1
        
        resultados[est].append(regret_acumulado_instante)

print("RESULTADOS FINALES (6 ANUNCIOS)")
plt.figure(figsize=(12, 7))
for est in estrategias:
    datos = np.array(resultados[est])
    media = np.mean(datos, axis=0)
    desviacion = np.std(datos, axis=0)
    
    plt.plot(media, label=est, linewidth=2.5)
    colores = {'AB_Testing': 'blue', 'Epsilon_Greedy': 'orange', 'Thompson_Sampling': 'green'}
    plt.fill_between(range(intentos), 
                 np.maximum(0, media - desviacion), 
                 media + desviacion, 
                 color=colores[est], alpha=0.15)
    
    print(f"Estrategia: {est:17} | Regret Medio Final: {media[-1]:.2f}")

plt.xlabel('Número de Usuarios (Tiempo t)', fontsize=14, fontweight='bold')
plt.ylabel('Regret Acumulado Medio', fontsize=14, fontweight='bold')
plt.title(f'Comparativa 6 Anuncios: Mayor complejidad de selección (Promedio de {num_experimentos} {"simulación" if num_experimentos == 1 else "simulaciones"})', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tick_params(labelsize=12)
plt.show()
