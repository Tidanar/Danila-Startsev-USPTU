from scipy import interpolate


def f(x_in):
    x = [0, 1, 2, 3, 4, 5]
    y = list(map(lambda a: a ** 3 + 7, x))
    return interpolate.splev(x_in, interpolate.splrep(x, y))


x = list(map(lambda a: 4 + a * 3 / 1000, range(1000)))
print(f(x))
