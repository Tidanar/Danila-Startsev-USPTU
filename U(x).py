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

def F0(x, a):
    return a[0] + a[1] * x + a[2] * x ** 2 + a[3] * x ** 3


def U0(dx, m, a, al):
    result = []
    for x in [xx/m for xx in range(m+1)]:
        result.append(sum(al[i] * F0(x, a[i]) for i in [0, 1]) + dx * sum(al[j] * F0(x, a[j]) for j in [2, 3]))
    return result


def integral(CC, dx, n, a):
    for i in range(4):
        for j in range(4):
            #C[i+4*n, j+4*n] = 4*a[i,2]*a[j,2]*dx + 6*(a[i,3]*a[j,2]+a[i,2]*a[j,3])*(dx**2) + 12*a[i,3]*a[j,3]*(dx**3)
            CC[i+4*n, j+4*n] = 4*a[i,2]*a[j,2]*dx + 6*(a[i,3]*a[j,2]+a[i,2]*a[j,3])*(dx**2) + 12*a[i,3]*a[j,3]*(dx**3)
    return CC

def U(m, x0, y0):
    F = [[1, 0, 0, 0],
         [1, 1, 1, 1],
         [0, 1, 0, 0],
         [0, 1, 2, 3]]

    E = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]

    A = np.linalg.solve(F, E).transpose()

    #C = np.zeros([len(x0) * 4 - 4, len(x0) * 4 - 4])
    CC = np.zeros([len(x0) * 5 - 4, len(x0) * 5 - 4])
    ZU = list(np.zeros(len(x0) * 4 - 4)) + y0
    for i in range(len(x0) - 1):
        CC = integral(CC, x0[i + 1] - x0[i], i, A)
    for i in range(len(x0) * 4 - 4):
        ii = i%4 if (i%4==0 or i%4==3) else (1 if i%4==2 else 2)
        CC[len(x0) * 4 - 4 + i // 4, i] = F0(x0[i // 4], A[ii])
        CC[len(x0) * 4 - 3 + i // 4, i] = F0(x0[i // 4 + 1], A[ii])
        CC[i, len(x0) * 4 - 4 + i // 4] = F0(x0[i // 4], A[ii])
        CC[i, len(x0) * 4 - 3 + i // 4] = F0(x0[i // 4 + 1], A[ii])

    beta_mu = list(np.linalg.solve(CC, ZU))
    beta0 = beta_mu[:len(x0) * 4 - 4]
    print(beta0)
    beta = beta0[:2]
    for i in range(len(x0) - 2):
        beta = beta + beta0[2 + 4*i:4 + 4*i]
    beta = beta + [beta0[-2]] + [beta0[-1]]

    S = np.zeros([len(x0) * 4 - 4, len(x0) * 2])
    for i in range(len(S)):
        for j in range(len(S[i])):
            if ((i % 4 == 0) and (j == i // 2)) or\
               ((i % 4 == 1) and (j - 2 == (i - 1) // 2)) or\
               ((i % 4 == 2) and (j - 1 == (i - 2) // 2)) or\
               ((i % 4 == 3) and (j - 3 == (i - 3) // 2)): S[i, j] = 1

    alfa_iter = S.dot(beta)
    # x, U_iter = [], []
    for i in range(len(x0) - 1):
        delta_x = x0[i + 1] - x0[i]
        alfa = alfa_iter[i * 4:i * 4 + 4]
        # x = x + [delta_x * xx/m + x0[i] for xx in range(m+1)]
        # U_iter = U_iter + U0(delta_x, m, A, alfa)
        plt.plot([delta_x * xx/m + x0[i] for xx in range(m+1)], U0(delta_x, m, A, alfa))
    # return x, U_iter


M = 100
k = 1 + int(5 * rnd()) + 2
x0 = [rnd() + i for i in range(k)]
y0 = [rnd() + i for i in range(k)]
U(M, x0, y0)
# x_iter, y_iter = U(M, x0, y0)
# plt.axis([x0[0], x0[-1], min(y_iter), max(y_iter)])
# plt.plot(x_iter, y_iter, 'k')
plt.plot(x0, y0, 'ro')
plt.show()
