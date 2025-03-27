import numpy as np
import matplotlib.pyplot as plt

M = np.array([89.46, 63.7, 49.59, 32.63, 23.81, 16.69])
H = np.linspace(0.5, 2.5, 5)
g = 9.81

for m, i in enumerate(M):
    for h, j in enumerate(H):
        E = m * g * h * 1e-3
        plt.loglog(E, i*j, "x")
        plt.text(E, i*j, f'M={m:.1f}, H={h:.1f}', fontsize=9, ha='right')

plt.show()
