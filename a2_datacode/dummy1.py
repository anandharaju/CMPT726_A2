from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

eta = [0.5, 0.3, 0.1, 0.05, 0.01]
sgd = [1.7536814899433457, 1.3575431443402122, 0.9717189004240064, 0.763719614859719, 0.3935292693729292]
gd = [0.252445, 0.277211, 0.361374, 0.426069, 0.523556]

plt.plot(eta, gd, label="Gradient Descent")
plt.plot(eta, sgd, label="Stochastic Gradient Descent")
plt.legend()
plt.ylabel('Negative log likelihood after 500 epochs')
plt.title('Comparison of GD and SGD')
plt.xlabel('ETA')
plt.show()



