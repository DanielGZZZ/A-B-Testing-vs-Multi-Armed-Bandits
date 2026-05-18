import numpy as np
import matplotlib.pyplot as plt

# Configuración del experimento
steps = 600 # Número de impresiones
p_best = 0.7  # Probabilidad de éxito del mejor anuncio
p_worst = 0.5 # Probabilidad de éxito del peor anuncio

# 1. Simulación A/B Test (Regret Lineal)
# El A/B test mantiene un error constante porque siempre explora al 50%
ab_regret = np.cumsum([0.5 * (p_best - p_worst) for _ in range(steps)])

# 2. Simulación MAB (Regret Logarítmico)
# El MAB aprende y deja de elegir el brazo malo, el error se aplana
mab_regret = np.cumsum([(p_best - p_worst) / (1 + 0.1 * t) for t in range(steps)])

# Visualización
plt.figure(figsize=(10, 6))
plt.plot(ab_regret, label='A/B Testing (Regret Lineal)', color='red', linestyle='--')
plt.plot(mab_regret, label='MAB (Regret Logarítmico)', color='green')

plt.title('Comparativa de Arrepentimiento Acumulado ($L_T$)')
plt.xlabel('Tiempo (Número de impresiones)')
plt.ylabel('Pérdida de Recompensa (Regret)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
