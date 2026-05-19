import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

plt.rcParams['font.family'] = 'serif'

escenarios = [
    { "pasos": 0, "clicks": 0, "alpha": 1, "beta": 1, "color": "#7F8C8D", "label": r"Inicial: $\alpha=1,\ \beta=1$"},
    { "pasos": 10, "clicks": 2, "alpha": 3, "beta": 9, "color": "#E74C3C", "label": r"10 imp. (2 clics): $\alpha=3,\ \beta=9$"},
    { "pasos": 100, "clicks": 24, "alpha": 25, "beta": 77, "color": "#3498DB", "label": r"100 imp. (24 clics): $\alpha=25,\ \beta=77$"},
    { "pasos": 1000, "clicks": 252, "alpha": 253, "beta": 749, "color": "#2ECC71", "label": r"1000 imp. (252 clics): $\alpha=253,\ \beta=749$"}
]

x = np.linspace(0, 0.6, 1000)

plt.figure(figsize=(8.5, 5))

for esc in escenarios:

    y = beta.pdf(x, esc["alpha"], esc["beta"])

    plt.plot(x, y, label=esc["label"], color=esc["color"], linewidth=2)
    plt.fill_between(x, y, color=esc["color"], alpha=0.08)

# Diseño
plt.title("Evolución de la Distribución Beta", fontsize=14, fontweight='bold')
plt.xlabel(r"Tasa de conversión estimada ($\theta$)")
plt.ylabel("Densidad")
plt.xlim(0, 0.5)
plt.ylim(bottom=0)
plt.grid(alpha=0.25)
plt.legend(fontsize=9, frameon=False, loc="upper right")

plt.tight_layout()
plt.show()
