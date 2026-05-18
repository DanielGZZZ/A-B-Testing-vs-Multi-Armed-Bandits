import numpy as np
import matplotlib.pyplot as plt

# Configuración
np.random.seed(42)
true_q = 0.65  # El valor real (q*) que el algoritmo no conoce
n_steps = 500
rewards = np.random.binomial(1, true_q, n_steps) # Simulamos éxitos/fracasos

# Cálculo de la media acumulada (Qt)
q_estimates = np.cumsum(rewards) / np.arange(1, n_steps + 1)

# Visualización
plt.figure(figsize=(10, 6))
plt.plot(q_estimates, label='Valor Estimado $Q_t(a)$', color='blue', linewidth=2)
plt.axhline(y=true_q, color='red', linestyle='--', label='Valor Real $q_*(a)$')

plt.title('Convergencia del Valor de Acción en el Tiempo')
plt.xlabel('Número de veces seleccionado $N_t(a)$')
plt.ylabel('Recompensa Media Estimada')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
