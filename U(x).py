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


def U(x):
    F = [[1, 0, 0, 0],
         [1, 1, 1, 1],
         [0, 1, 0, 0],
         [0, 1, 2, 3]]

    E = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]

    A = np.linalg.solve(F, E)
    alfa = [1., 0.5, 0.78, 0.34, 0.22]
    U_iter = []

    return U_iter


plt.axis([0, 1])
x_iter = [x / 100 for x in range(0, 100)]
y_iter = U(x_iter)
plt.plot(x_iter, y_iter, 'k')
plt.show()
