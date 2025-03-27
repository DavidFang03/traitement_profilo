import matplotlib. pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import utilit

data = np.loadtxt("test.txt")
x = data[:, 0]
y = data[:, 1]

fig, ax = plt.subplots()
line = ax.plot([0], [0], 'o', label='data')

popt, pcov = curve_fit(utilit.hyperbolic, x, y, p0=[
                       np.min(y), 0, 0, x[np.argmin(y)]])

plt.plot(x, y, 'o', label='data')
plt.plot(x, utilit.hyperbolic(x, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f, x0=%5.3f' % tuple(popt))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data points')
plt.legend()
plt.show()
