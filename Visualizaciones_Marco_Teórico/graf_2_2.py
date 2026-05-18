import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

x = np.linspace(0.02, 0.18, 1000)

# Anuncio A: pico en x=0.10, altura 40
y_A = beta.pdf(x, 91, 811)

# Anuncio B: pico en x=0.08, altura 14
y_B = beta.pdf(x, 9, 93)

color_A = '#1f4e8c'
color_B = '#d97c2b'

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(x, y_A, color=color_A, linewidth=2, label='Anuncio A (Opción conocida)')
ax.plot(x, y_B, color=color_B, linewidth=2, label='Anuncio B (Opción incierta)')
ax.fill_between(x, y_A, alpha=0.15, color=color_A)
ax.fill_between(x, y_B, alpha=0.15, color=color_B)

ax.axvline(x=0.10, color=color_A, linestyle='--', linewidth=1.2, alpha=0.8)
ax.axvline(x=0.08, color=color_B, linestyle='--', linewidth=1.2, alpha=0.8)

ax.annotate('Explotar A:\nResultado probable', xy=(0.107, 35), fontsize=8.5, color=color_A, fontweight='bold')
ax.annotate('Explorar B:\nPotencial oculto',   xy=(0.044, 15), fontsize=8.5, color=color_B, fontweight='bold')

ax.set_xlabel('Tasa de Conversión Estimada (CTR / CVR)', fontsize=10)
ax.set_ylabel('Densidad de Probabilidad', fontsize=10)
ax.set_title('Visualización de la Incertidumbre: Dilema Exploración vs. Explotación', fontsize=11)
ax.set_xlim(0.02, 0.18)
ax.set_ylim(0, 42)
ax.set_xticks(np.arange(0.02, 0.19, 0.02))
ax.grid(True, linestyle='-', alpha=0.4, color='#cccccc')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()
