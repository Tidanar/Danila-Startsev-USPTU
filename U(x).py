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


def F0(x, a):
    return a[0] + a[1] * x + a[2] * x ** 2 + a[3] * x ** 3


def U0(dx, n, a, al):
    result = []
    for x in [xx/n for xx in range(n+1)]:
        result.append(sum(al[i] * F0(x, a[i]) for i in [0, 2]) + dx * sum(al[j] * F0(x, a[j]) for j in [1, 3]))
    return result


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
    A = np.array([A[0], A[2], A[1], A[3]])

    C = np.zeros([4, 4])
    for i in range(4):
        for j in range(4):
            C[i, j] = 4*A[i, 2]*A[j, 2] + 6*(A[i, 3]*A[j, 2]+A[i, 2]*A[j, 3]) + 12*A[i, 3]*A[j, 3]

    core = np.zeros([4*m, 4*m])
    for i in range(m):
        core[4 * i:4 * i + 4, 4 * i:4 * i + 4] = C / (1/m) ** 3

    S = np.zeros([4 * m, 2 * (m + 1)])
    for i in range(len(S)):
        if i % 4 == 0:
            S[i, i // 2] = 1
        elif i % 4 == 1:
            S[i, (i - 1) // 2 + 1] = 1
        elif i % 4 == 2:
            S[i, (i - 2) // 2 + 2] = 1
        elif i % 4 == 3:
            S[i, (i - 3) // 2 + 3] = 1

    core = (S.transpose().dot(core)).dot(S)
    ZU = np.hstack((np.zeros(2 * (m + 1)), y0))
    fi = np.zeros([len(x0), 2 * (m + 1)])
    for i in range(len(x0)):
        fi[i, 2 * (int(x0[i] * m)) + np.array([0, 1, 2, 3])] = [F0(x0[i], A[j]) for j in [0, 1, 2, 3]]

    CC = np.zeros([len(x0) + 4 * m, len(x0) + 4 * m])
    CC = np.vstack((np.hstack((core, fi.transpose())), np.hstack((fi, np.zeros([len(x0), len(x0)])))))
    beta_mu = list(np.linalg.solve(CC, ZU))
    beta = beta_mu[:2 * (m + 1)]
    alfa_iter = S.dot(beta)

    x_iter, U_iter = [], []
    delta_x, n = 1 / m, 10
    for i in range(m):
        alfa = alfa_iter[i * 4:i * 4 + 4]
        x_iter = x_iter + [delta_x * xx / n + i / m for xx in range(n + 1)]
        U_iter = U_iter + U0(delta_x, n, A, alfa)
    return x_iter, U_iter


M, k, step = 200, 10, 0.001
x0 = list(np.sort(np.random.rand(k)))
y0 = np.random.rand(k)
x_iter, y_iter = U(M, x0, y0)
plt.plot(x_iter, y_iter, 'b')

b_cs, c_cs, d_cs = cs_coefficients(x0, y0)
x_iter, y_iter = [], []
for i in range(len(x0) - 1):
    x_iter = np.arange(x0[i], x0[i+1], step)
    y_iter = [y0[i] + b_cs[i] * (x - x0[i]) + c_cs[i] * (x - x0[i]) ** 2 + d_cs[i] * (x - x0[i]) ** 3 for x in x_iter]
    plt.plot(x_iter, y_iter, 'r')

plt.plot(x0, y0, 'ko')
plt.show()
