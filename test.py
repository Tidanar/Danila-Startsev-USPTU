import numpy as np
import matplotlib.pyplot as plt


def cs_coefficients(x, y):
    x = np.array(x)
    y = np.array(y)
    size = len(x)
    delta_x = np.diff(x)
    delta_y = np.diff(y)
    A = np.zeros(shape=(size, size))
    b = np.zeros(shape=(size, 1))
    A[0, 0] = 1
    A[-1, -1] = 1
    for i in range(1, size - 1):
        A[i, i - 1] = delta_x[i - 1]
        A[i, i + 1] = delta_x[i]
        A[i, i] = 2 * (delta_x[i - 1] + delta_x[i])
        b[i, 0] = 3 * (delta_y[i] / delta_x[i] - delta_y[i - 1] / delta_x[i - 1])
    c = np.linalg.solve(A, b)   # решение СЛАУ
    d = np.zeros(shape=(size - 1, 1))
    b = np.zeros(shape=(size - 1, 1))
    for i in range(0, len(d)):
        d[i] = (c[i + 1] - c[i]) / (3 * delta_x[i])
        b[i] = (delta_y[i] / delta_x[i]) - (delta_x[i] / 3) * (2 * c[i] + c[i + 1])

    return b.squeeze(), c.squeeze(), d.squeeze()


step = 0.001
k = 10
x0 = list(np.sort(np.random.rand(k)))
y0 = list(np.random.rand(k))
b_cs, c_cs, d_cs = cs_coefficients(x0, y0)
x_iter, y_iter = [], []
for i in range(len(x0) - 1):
    x_iter = np.arange(x0[i], x0[i+1], step)
    y_iter = [y0[i] + b_cs[i] * (x - x0[i]) + c_cs[i] * (x - x0[i]) ** 2 + d_cs[i] * (x - x0[i]) ** 3 for x in x_iter]
    plt.plot(x_iter, y_iter, 'k')
plt.plot(x0, y0, 'ro')
plt.show()
