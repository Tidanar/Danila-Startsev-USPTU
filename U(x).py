"""
P(x) = a0 + a1 * x + a2 * x^2 + a3 * x^3
P'(x) = a1 + 2 * a2 * x + 3 * a3 * x^2
F00 = P(0) = a0
F10 = P(1) = a0 + a1 + a2 + a3
F01 = P'(0) = a1
F11 = P'(1) = a1 + 2 * a2 + 3 * a3
"""

import numpy as np
import matplotlib.pyplot as plt
from random import random as rnd

def F00(x, a):
    return a[0] + a[1] * x + a[2] * x ** 2 + a[3] * x ** 3


def F10(x, a):
    return a[0] + a[1] * x + a[2] * x ** 2 + a[3] * x ** 3


def F01(x, a):
    return a[1] + 2 * a[2] * x + 3 * a[3] * x ** 2


def F11(x, a):
    return a[1] + 2 * a[2] * x + 3 * a[3] * x ** 2


def U0(beta, m, a, alfa):
    result = []
    for x in [xx/m for xx in range(m+1)]:
        result = result + [alfa[0] * F00(x, a[0]) + alfa[1] * F01(x, a[1]) + beta * alfa[2] * F10(x, a[2]) + beta * alfa[3] * F11(x, a[3])]
    return result


def U(m, x0):
    F = [[1, 0, 0, 0],
         [1, 1, 1, 1],
         [0, 1, 0, 0],
         [0, 1, 2, 3]]

    E = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]

    A = np.linalg.solve(F, E).transpose()
    #x, U_iter = [], []
    for i in range(len(x0) - 1):
        alfa = [rnd() + 0.1, rnd() + 0.1, rnd() + 0.1, rnd() + 0.1]
        delta_x = x0[i + 1] - x0[i]
        #x = x + [delta_x * xx/m + x0[i] for xx in range(m+1)]
        #U_iter = U_iter + U0(delta_x, m, A, alfa)
        plt.plot([delta_x * xx/m + x0[i] for xx in range(m+1)], U0(delta_x, m, A, alfa))
    #return x, U_iter


M = 100
k = 1 + int(5 * rnd()) + 3
x0 = [rnd() + i for i in range(k)]
U(M, x0)
#x_iter, y_iter = U(M, x0)
#plt.axis([x0[0], x0[-1], min(y_iter), max(y_iter)])
#plt.plot(x_iter, y_iter, 'k')
#plt.plot(x0, [(min(y_iter)+max(y_iter))/2] * len(x0), 'ro')
plt.show()
