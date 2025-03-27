from tools_ThreeD import filter0, extract_bottom_3D_Profile
import numpy as np
import timeit

X1 = np.random.rand(100)
X2 = np.random.rand(100)
Y = np.random.rand(100, 100)

# Mesurer le temps d'exécution de filter0
time_filter0 = timeit.timeit(
    "filter0(X1, X2, Y)", globals=globals(), number=100)

# Mesurer le temps d'exécution de extract_bottom_3D_Profile
time_extract_bottom = timeit.timeit(
    "extract_bottom_3D_Profile(X1, X2, Y)", globals=globals(), number=100)

print(f"Temps d'exécution de filter0: {time_filter0} secondes")
print(
    f"Temps d'exécution de extract_bottom_3D_Profile: {time_extract_bottom} secondes")


print()