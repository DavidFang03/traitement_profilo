import matplotlib.pyplot as plt
import numpy as np

# Activer le backend Qt5
plt.switch_backend('Qt5Agg')

# Créer un espace de travail avec un pic gaussien
x = np.arange(20)
y0 = 10. + 50 * np.exp(-(x - 10)**2 / 20)

for i in range(4):
    fig, ax = plt.subplots()
    ax.plot(x, y0)  # tracer la courbe initiale en noir

figs = list(map(plt.figure, plt.get_fignums()))

a = 2000
b = 500

# Déplacer les fenêtres
figs[0].canvas.manager.window.move(0, 0)
figs[1].canvas.manager.window.move(a, 0)
figs[2].canvas.manager.window.move(0, b)
figs[3].canvas.manager.window.move(a, b)

plt.show()
