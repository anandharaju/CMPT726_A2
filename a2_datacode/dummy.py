import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
from matplotlib import cm
import random


def func1(x1, x2):
    return 6 + (2 * np.power(x1, 2)) + (2 * np.power(x2, 2))


def func2(x1, x2):
    return 8 + (0 * np.power(x1, 2)) + (0 * np.power(x2, 2))


seed = 1
samples = 250

x1 = np.linspace(-seed, seed, samples)
x2 = np.linspace(-seed, seed, samples)

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
# x1 = x2 = np.arange(-3.0, 3.0, 0.25)
X1, X2 = np.meshgrid(x1, x2)

y1 = np.array(func1(np.ravel(X1), np.ravel(X2)))
Y1 = y1.reshape(X1.shape)

y2 = np.array(func2(np.ravel(X1), np.ravel(X2)))
Y2 = y2.reshape(X1.shape)

sub1 = (Y1 >= Y2).astype(int)
sub2 = (Y2 >= Y1).astype(int)

Y1_col = cm.jet(sub1/-2.2 + 0.90)
Y2_col = cm.jet(sub2/-2.2 + 0.90)

# Y1_col = cm.jet(Y1)
# Y2_col = cm.jet(Y2)

ax.plot_surface(X1, X2, Y1, facecolors=Y1_col, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.5)
ax.plot_surface(X1, X2, Y2, facecolors=Y2_col, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.5)

# for i in range(0, samples):
#    ax.plot_surface(X1[i], X2[i], Y1[i], 'green' if y1[i] > y2[i] else 'red')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('yk(X)')


plt.savefig("D:\\00_SFU\\00_Graduate_Courses\\00_CMPT726_ML\\Assignments\\2\\231.png")
plt.show()
