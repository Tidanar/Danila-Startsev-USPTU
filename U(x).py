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

def F(x, a):
    return a[0] + a[1] * x + a[2] * x ** 2 + a[3] * x ** 3


def U0(dx, m, a, al):
    result = []
    for x in [xx/m for xx in range(m+1)]:
        result.append(sum(al[i] * F(x, a[i]) for i in [0, 1]) + dx * sum(al[j] * F(x, a[j]) for j in [2, 3]))
    return result


def integral(C, dx, n, a):
    for i in range(4):
        for j in range(4):
            C[i+4*n, j+4*n] = 4*a[i,2]*a[j,2]*dx + 6*(a[i,3]*a[j,2]+a[i,2]*a[j,3])*(dx**2) + 12*a[i,3]*a[j,3]*(dx**3)
    return C


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
    beta = [rnd() for _ in range(len(x0) * 2)]
    S = np.zeros([len(x0) * 4 - 4, len(beta)])
    for i in range(len(S)):
        for j in range(len(S[i])):
            if ((i % 4 == 0) and (j == i // 2)) or\
               ((i % 4 == 1) and (j - 2 == (i - 1) // 2)) or\
               ((i % 4 == 2) and (j - 1 == (i - 2) // 2)) or\
               ((i % 4 == 3) and (j - 3 == (i - 3) // 2)): S[i, j] = 1
    alfa_iter = S.dot(beta)
    C = np.zeros([len(x0) * 4 - 4, len(x0) * 4 - 4])
    # x, U_iter = [], []
    for i in range(len(x0) - 1):
        alfa = alfa_iter[i * 4:i * 4 + 4]
        delta_x = x0[i + 1] - x0[i]
        C = integral(C, delta_x, i, A)
        # x = x + [delta_x * xx/m + x0[i] for xx in range(m+1)]
        # U_iter = U_iter + U0(delta_x, m, A, alfa)
        plt.plot([delta_x * xx/m + x0[i] for xx in range(m+1)], U0(delta_x, m, A, alfa))
    # return x, U_iter


M = 100
k = 1 + int(5 * rnd()) + 3
x0 = [rnd() + i for i in range(k)]
U(M, x0)
# x_iter, y_iter = U(M, x0)
# plt.axis([x0[0], x0[-1], min(y_iter), max(y_iter)])
# plt.plot(x_iter, y_iter, 'k')
# plt.plot(x0, [(min(y_iter)+max(y_iter))/2] * len(x0), 'ro')
plt.show()
