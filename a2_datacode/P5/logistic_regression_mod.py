#!/usr/bin/env python
# Run logistic regression training.
import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2
import time

st = time.time()
# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
#  = 0.5
etas = [0.5, 0.3, 0.1, 0.05, 0.01]
eta_vs_error = dict()

# Load data.
data = np.genfromtxt('data.txt')
# Data matrix, with column of ones at end.
X = data[:, 0:3]
# Target values, 0 for class 1, 1 for class 2.
t = data[:, 3]

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

final_iter = None

for eta in etas:
    # Initialize w.
    w = np.array([0.1, 0, 0])
    # Error values over all iterations.
    e_all = []
    for iter in range(0, max_iter):
        final_iter = iter

        # Compute output using current w on all data X.
        y = sps.expit(np.dot(X, w))

        # e is the error, negative log-likelihood (Eqn 4.90)
        e = -np.mean(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))

        # Add this error to the end of error vector.
        e_all.append(e)

        # Gradient of the error, using Eqn 4.91
        grad_e = np.mean(np.multiply((y - t), X.T), axis=1)

        # Update w, *subtracting* a step in the error derivative since we're minimizing
        w_old = w
        w = w - eta * grad_e

        # Plot current separator and data.  Useful for interactive mode / debugging.
        '''plt.figure(DATA_FIG)
        plt.clf()
        plt.plot(X1[:, 0], X1[:, 1], 'b.')
        plt.plot(X2[:, 0], X2[:, 1], 'g.')
        a2.draw_sep(w)
        plt.axis([-5, 15, -10, 10])'''

        # Add next step of separator in m-b space.
        '''plt.figure(SI_FIG)
        a2.plot_mb(w, w_old)'''

        # Print some information.
        # print('epoch {0:d}, negative log-likelihood {1:.6f}, w={2}'.format(iter, e, w.T))
        # Stop iterating if error doesn't change more than tol.
        if iter > 0:
            if np.absolute(e - e_all[iter - 1]) < tol:
                break
    eta_vs_error[eta] = e_all
    print('Epoch: {0}\t\teta: {1}\t\tnegative log-likelihood: {2:.6f}'.format(final_iter, eta, e_all[-1]))

et = time.time()

# Plot error over iterations
plt.figure(5, figsize=(8.5, 6))
for eta in etas:
    plt.plot(eta_vs_error[eta])
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
plt.legend(etas)

'''
  eta_vs_error[eta] = e_all[-1]
plt.figure("eta_vs_error", figsize=(8.5, 6))
plt.plot(eta_vs_error.keys(), eta_vs_error.values())
plt.ylabel('Epoch')
plt.title('Training logistic regression - ETAs')
plt.xlabel('Negative log likelihood')
'''
plt.show()

print("Time Taken: ", et-st)
