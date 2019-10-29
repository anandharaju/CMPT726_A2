import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

seed = 1
samples = 500

x1 = np.linspace(-seed, seed, samples)
x2 = np.linspace(-seed, seed, samples)

y1 = 6 + (2 * np.power(x1, 2)) + (2 * np.power(x2, 2))
y2 = 8 + (0 * np.power(x1, 2)) + (0 * np.power(x2, 2))

# Question 2.1
plt.plot(y1, 'green', label="y1(x)")
plt.xlabel("x")
plt.ylabel("y1(x)")
plt.legend()
plt.show()

# Question 2.2
plt.plot(y2, 'red', label="y2(x)")
plt.xlabel("x")
plt.ylabel("y2(x)")
plt.legend()
plt.show()

# Question 2.3
plt.plot(y1, 'green', label="y1(x)")
plt.xlabel("x")
plt.ylabel("y(x)")

plt.plot(y2, 'red', label="y2(x)")
plt.xlabel("x")
plt.ylabel("y(x)")

for i in range(0, samples):
    m, n = [i, i], [y1[i], y2[i]]
    plt.plot(m, n, 'green' if y1[i] > y2[i] else 'red', linestyle=':', linewidth=0.75)

plt.text(samples / 50, 8.1, "argmaxk yk(x) = y1(x)", rotation=-60)
plt.text(samples / 2.35, 7.1, "argmaxk yk(x) = y2(x)")
plt.text(samples / 1.125, 8.1, "argmaxk yk(x) = y1(x)", rotation=60)

plt.legend()
plt.show()
