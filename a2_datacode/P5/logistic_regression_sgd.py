#!/usr/bin/env python
# Run logistic regression training.
import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2
import time
import random

st = time.time()

# Step size for gradient descent.
#  = 0.5
etas = [0.5, 0.3, 0.1, 0.05, 0.01]
eta_vs_error = dict()
errors = np.zeros(len(etas))

# Load data.
data = np.genfromtxt('data.txt')
# Data matrix, with column of ones at end.
X = data[:, 0:3]
# Target values, 0 for class 1, 1 for class 2.
t = data[:, 3]

np.random.seed(128) # 4

randomize = np.arange(len(X))
np.random.shuffle(randomize)

X = X[randomize]
t = t[randomize]

# For plotting data
class1 = np.where(t == 0)
X1 = X[class1]
class2 = np.where(t == 1)
X2 = X[class2]

DATA_FIG = 1

# Set up the slope-intercept figure
'''SI_FIG = 2
plt.figure(SI_FIG, figsize=(8.5, 6))
plt.rcParams.update({'font.size': 15})
plt.title('Separator in slope-intercept space')
plt.xlabel('slope')
plt.ylabel('intercept')
plt.axis([-5, 5, -10, 0])'''
# np.random.seed(8)  # 2 7
num_of_samples = len(X)

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
data_len = 200
tol = 0.00001
trials = 500

last_iter = np.zeros(trials)

e_all_avg = 0
for eta in etas:
    w = np.array([0.1, 0, 0])
    e_all = []
    for trial in range(0, trials):
        for i in range(0, data_len):

            y = sps.expit(np.dot(X[i], w))
            grad_e = np.multiply((y - t[i]), X[i].T)
            # Update w, *subtracting* a step in the error derivative since we're minimizing
            w_old = w
            w = w - eta * grad_e

        # Compute output using current w on all data X.
        y = sps.expit(np.dot(X, w))
        e = -np.mean(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))
        e_all.append(e)

        if trial > 0:
            if np.absolute(e - e_all[trial - 1]) < tol:
                break
    print(eta, e_all[-1])
    plt.plot(e_all)


'''
# Plot error over iterations
plt.figure(5, figsize=(8.5, 6))
for eta in etas:
    plt.plot(eta_vs_error[eta])
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression (On a random sample per iteration)')
plt.xlabel('Iteration (Total Iterations - '+str(max_iter)+')')
plt.legend(etas)
'''


'''
et = time.time()

# plt.figure("eta_vs_error", figsize=(8.5, 6))
for i, eta in enumerate(etas):
    errors[i] = eta_vs_error[eta]

plt.plot(etas, errors)
plt.ylabel('Average Negative log-likelihood')
plt.title('SGD based logistic regression training - ETAs')
plt.xlabel('ETA Value')
plt.yscale('log')
plt.xticks(etas)
plt.legend(["ETA Values"])
print("Time Taken: ", (et - st)/trials)
'''

plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch [1 Epoch = 200 Iterations]')
plt.legend(etas)
plt.show()
plt.show()
