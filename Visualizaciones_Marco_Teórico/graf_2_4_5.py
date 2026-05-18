import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

plt.figure(figsize=(10, 6))
x = np.linspace(0, 1, 1000)  # Eje x: tasa de éxito (0 a 1)

# Acción 1 (Azul): ALTA CERTIDUMBRE - MUY BUENA
# Beta(70, 30): 70 éxitos, 30 fracasos. Media = 0.7, MUY ESTRECHA.
# Anuncio con muchos datos que sabemos que es excelente.
a1, b1 = 70, 30
plt.plot(x, beta.pdf(x, a1, b1), label='Acción 1 (Azul): Beta(70,30)', color='#1f77b4', lw=3)
plt.fill_between(x, beta.pdf(x, a1, b1), color='#1f77b4', alpha=0.15)

# Acción 2 (Roja): ALTA INCERTIDUMBRE - INTERMEDIA
# Beta(5, 5): 5 éxitos, 5 fracasos. Media = 0.5, MUY ANCHA.
# Anuncio con pocos datos, su valor real es un misterio.
a2, b2 = 2, 2
plt.plot(x, beta.pdf(x, a2, b2), label='Acción 2 (Roja): Beta(2,2)', color='#d62728', lw=3)
plt.fill_between(x, beta.pdf(x, a2, b2), color='#d62728', alpha=0.15)

# Acción 3 (Verde): ALTA CERTIDUMBRE - MALA
# Beta(30, 70): 30 éxitos, 70 fracasos. Media = 0.3, MUY ESTRECHA.
# Anuncio con muchos datos que sabemos que es pobre.
a3, b3 = 30, 70
plt.plot(x, beta.pdf(x, a3, b3), label='Acción 3 (Verde): Beta(30,70)', color='#2ca02c', lw=3)
plt.fill_between(x, beta.pdf(x, a3, b3), color='#2ca02c', alpha=0.15)

plt.title('Actualización de la Distribución Beta en Thompson Sampling', fontsize=16, fontweight='bold')
plt.xlabel('Tasa de Éxito Potencial ($theta$)', fontsize=14)
plt.ylabel('Densidad de Probabilidad', fontsize=14)
plt.xlim(0, 1)
plt.ylim(0, 10)
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=11, loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
