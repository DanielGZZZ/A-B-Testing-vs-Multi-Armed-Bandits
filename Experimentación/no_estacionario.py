import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class EntornoPublicitarioNoEstacionario:
    def __init__(self, cambio_tiempo=20000):
        self.cambio_tiempo = cambio_tiempo
        
        self.tasas_iniciales = {
            'Anuncio_A1': 0.04,
            'Anuncio_A2': 0.038,
            'Anuncio_B1': 0.05,
            'Anuncio_B2': 0.042,
            'Anuncio_C1': 0.02,
            'Anuncio_C2': 0.018
        }
        
        self.tasas_actuales = self.tasas_iniciales.copy()
        self.acciones = list(self.tasas_iniciales.keys())

    def actualizar_tasas(self, t):
        if t == self.cambio_tiempo:
            nuevas = self.tasas_actuales.copy()
            
            # Cambio de tendencia total, el ganador se convierte en el perdedor y viceversa
            nuevas['Anuncio_B1'], nuevas['Anuncio_C1'] = (
                self.tasas_actuales['Anuncio_C1'],
                self.tasas_actuales['Anuncio_B1']
            )
            
            # Ligeras variaciones en el resto de anuncios
            nuevas['Anuncio_A1'] += np.random.uniform(-0.005, 0.005)
            nuevas['Anuncio_A2'] += np.random.uniform(-0.005, 0.005)
            nuevas['Anuncio_B2'] += np.random.uniform(-0.005, 0.005)
            nuevas['Anuncio_C2'] += np.random.uniform(-0.005, 0.005)
            
            self.tasas_actuales = nuevas

    def generar_respuesta_usuario(self, accion, t):
        """Simula la respuesta del usuario a un anuncio haciendo clic o no usando una distribución de Bernoulli.
        Args:            
            accion (str): El anuncio mostrado al usuario.
            t (int): El número de intento actual, necesario para actualizar las tasas en el momento del cambio.
        Returns:            bool: True si el usuario hizo clic, False en caso contrario."""
        self.actualizar_tasas(t)
        return np.random.rand() < self.tasas_actuales[accion]

    def mejor_tasa_actual(self):
        return max(self.tasas_actuales.values())


# THOMPSON SAMPLING 
def thompson_sampling_step(alphas, betas):
    """Realiza un paso de Thompson Sampling para elegir un anuncio.
    Args:
        alphas (dict): Parámetros alpha de las distribuciones Beta para cada anuncio.
        betas (dict): Parámetros beta de las distribuciones Beta para cada anuncio.
    Returns:
        str: El anuncio elegido para mostrar al usuario.
    """
    muestras = {}
    for a in alphas.keys():
        muestras[a] = np.random.beta(alphas[a], betas[a])
    return max(muestras, key=muestras.get)


# A/B TESTING 
ganador_ab = None

def ab_testing_step(t, alphas, betas, periodo_exploracion=1500):
    """Realiza un paso de A/B Testing con un periodo de exploración fijo.
    Args:
        t (int): El número de intento actual.
        alphas (dict): Parámetros alpha de las distribuciones Beta para cada anuncio.
        betas (dict): Parámetros beta de las distribuciones Beta para cada anuncio.
        periodo_exploracion (int): Número de intentos dedicados a la exploración pura.
    Returns:
        str: El anuncio elegido para mostrar al usuario.
    """
    global ganador_ab
    acciones = list(alphas.keys())
    
    if t < periodo_exploracion:
        return acciones[t % len(acciones)]
    elif t == periodo_exploracion:
        mejor = None
        mejor_score = -1
        
        for a in acciones:
            score = alphas[a] / (alphas[a] + betas[a])
            if score > mejor_score:
                mejor_score = score
                mejor = a
        
        ganador_ab = mejor
        return ganador_ab
    else:
        return ganador_ab

# EPSILON-GREEDY
def epsilon_greedy_step(alphas, betas, epsilon=0.1):
    """Realiza un paso de Epsilon-Greedy para elegir un anuncio.
    Args:
        alphas (dict): Parámetros alpha de las distribuciones Beta para cada anuncio.
        betas (dict): Parámetros beta de las distribuciones Beta para cada anuncio.
        epsilon (float): Probabilidad de explorar (elegir un anuncio aleatorio).
    Returns:
        str: El anuncio elegido para mostrar al usuario."""
    if np.random.rand() < epsilon:
        return np.random.choice(list(alphas.keys()))
    return max(alphas, key=lambda a: alphas[a] / (alphas[a] + betas[a]))


# Ventana deslizante para no estacionariedad
class VentanaDeslizante:
    """Implementa una ventana deslizante para mantener un historial limitado de interacciones y adaptarse a cambios en el entorno.
     Args:
        size (int): El tamaño de la ventana, es decir, cuántas interacciones recientes se consideran para actualizar las creencias sobre cada anuncio.
        acciones (list): Lista de anuncios disponibles en el entorno."""
    def __init__(self, size, acciones):
        self.size = size
        self.historial = {a: deque() for a in acciones} 
        self.sumas = {a: 0 for a in acciones}  

    def actualizar(self, accion, recompensa):
        cola = self.historial[accion]
        
        if len(cola) == self.size:
            eliminado = cola.popleft()
            self.sumas[accion] -= eliminado  # quitar viejo
        
        cola.append(recompensa)
        self.sumas[accion] += recompensa  # añadir nuevo

    def get_alphas_betas(self):
        alphas = {}
        betas = {}
        
        for a in self.historial:
            suma = self.sumas[a]
            n = len(self.historial[a])
            
            alphas[a] = 1 + suma
            betas[a] = 1 + n - suma
            
        return alphas, betas

# Simulación
intentos = 50000
num_experimentos = 100
window_size = 1000

estrategias = ['AB_Testing', 'Epsilon_Greedy', 'Thompson_Sampling']
resultados = {est: [] for est in estrategias}
elecciones_ts_final = []  

for est in estrategias:
    for exp_id in range(num_experimentos):
        ganador_ab = None
        entorno = EntornoPublicitarioNoEstacionario()
        ventana = VentanaDeslizante(window_size, entorno.acciones)
        
        regret_total = 0
        regret_acumulado_instante = []
        elecciones_actuales = [] 
        
        for t in range(intentos):
            alphas, betas = ventana.get_alphas_betas()
            
            if est == 'AB_Testing':
                a_t = ab_testing_step(t, alphas, betas)
            elif est == 'Epsilon_Greedy':
                a_t = epsilon_greedy_step(alphas, betas)
            else:
                a_t = thompson_sampling_step(alphas, betas)
            
            # Guardar elección
            elecciones_actuales.append(a_t)
            
            exito = entorno.generar_respuesta_usuario(a_t, t)
            ventana.actualizar(a_t, int(exito))
            
            mejor_actual = entorno.mejor_tasa_actual()
            regret_total += (mejor_actual - entorno.tasas_actuales[a_t])
            regret_acumulado_instante.append(regret_total)
        
        resultados[est].append(regret_acumulado_instante)
        
        if est == 'Thompson_Sampling' and exp_id == num_experimentos - 1:
            elecciones_ts_final = elecciones_actuales

# Gráfica 1 - Regret acumulado medio con desviación estándar
plt.figure(figsize=(12, 7))

for est in estrategias:
    datos = np.array(resultados[est])
    media = np.mean(datos, axis=0)
    desviacion = np.std(datos, axis=0)
    
    plt.plot(media, label=est, linewidth=2.5)
    plt.fill_between(range(intentos),
                     np.maximum(0, media - desviacion),
                     media + desviacion,
                     alpha=0.15)

    print(f"Estrategia: {est:17} | Regret Medio Final: {media[-1]:.2f}")

plt.axvline(20000, color='red', linestyle='--', label='Cambio de entorno')
plt.xlabel('Número de Usuarios (Tiempo t)', fontsize=14, fontweight='bold')
plt.ylabel('Regret Acumulado Medio', fontsize=14, fontweight='bold')
plt.title(f'Comparativa 6 Anuncios: Escenario NO Estacionario (Promedio de {num_experimentos} {"simulación" if num_experimentos == 1 else "simulaciones"})', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Grafica 2 - Zoom en la transición cultural
plt.figure(figsize=(12, 6))

for est in estrategias:
    datos = np.array(resultados[est])
    media = np.mean(datos, axis=0)
    plt.plot(media, label=est, linewidth=3)

t_inicio, t_fin = 17000, 29000
plt.xlim(t_inicio, t_fin)

# Ajuste dinámico del eje Y para que se vea el quiebro de las líneas
todos_los_datos_zoom = [np.mean(resultados[e], axis=0)[t_inicio:t_fin] for e in estrategias]
plt.ylim(np.min(todos_los_datos_zoom) * 0.95, np.max(todos_los_datos_zoom) * 1.05)

plt.axvline(20000, color='red', linestyle='--', linewidth=2)

val_ts_20k = np.mean(resultados['Thompson_Sampling'], axis=0)[20000]
val_ts_recup = np.mean(resultados['Thompson_Sampling'], axis=0)[25000]

plt.xlabel('Número de Usuarios (Tiempo t)', fontsize=12, fontweight='bold')
plt.ylabel('Regret Acumulado Medio', fontsize=12, fontweight='bold')
plt.title('Detalle Cambio Entorno (Transición Cultural)', fontsize=14)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.show()


# Gráfica 3 - Dinámica de selección de anuncios (Thompson Sampling)
def plot_arm_selection(elecciones, nombres_brazos, t_cambio):
    intervalo = 500 
    pasos = range(0, len(elecciones), intervalo)
    
    # Calculamos porcentajes de elección por cada bloque de 500 usuarios
    proporciones = {accion: [] for accion in nombres_brazos}
    for p in pasos:
        bloque = elecciones[p : p + intervalo]
        for accion in nombres_brazos:
            cantidad = bloque.count(accion)
            proporciones[accion].append((cantidad / len(bloque)) * 100)

    plt.figure(figsize=(12, 6))
    paleta = plt.get_cmap('tab10', len(nombres_brazos))
    
    plt.stackplot(pasos, proporciones.values(), 
                  labels=proporciones.keys(), 
                  alpha=0.8, colors=paleta.colors)
    
    plt.axvline(t_cambio, color='white', linestyle='--', linewidth=2)
    
    plt.title('Dinámica de Selección de Anuncios (Thompson Sampling)', fontsize=14)
    plt.xlabel('Número de Usuarios (Tiempo t)', fontsize=12, fontweight='bold')
    plt.ylabel('% de veces que se muestra cada anuncio', fontsize=12, fontweight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Anuncios")
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.show()

plot_arm_selection(elecciones_ts_final, entorno.acciones, 20000)
