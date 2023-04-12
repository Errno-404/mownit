# works with clamped somehow

import numpy as np
from matplotlib import pyplot as plt
from enum import Enum

A = -np.pi
B = 2 * np.pi
POINTS = 500


class Boundary(Enum):
    NATURAL = 1
    CLAMPED = 2


def f(x):
    k = 3
    m = 3
    return np.exp(-k * np.sin(m * x)) + k * np.sin(m * x) - 1


def df(x):
    k = 3
    m = 3
    return k * m * np.cos(m * x) * (1 - np.exp(-k * np.sin(m * x)))



def create_equidistant_nodes(a, b, n):
    return np.linspace(a, b, n)


def get_h(x):
    n = len(x)
    return [x[i + 1] - x[i] for i in range(n - 1)]


# if n is equal to node count, then we have n-1 splines!
def spline(n, h, x, y, r=Boundary.NATURAL):
    m = n - 1
    matrix = [[0 for _ in range(m)] for _ in range(m)]
    rval = [0 for _ in range(m)]
    for i in range(0, m - 1):
        matrix[i][i] = 1
        matrix[i][i + 1] = 1
        rval[i] = 2 * (y[i + 1] - y[i]) / h[i]

    # boundary here:

    if r == Boundary.NATURAL:
        matrix[m - 1][m - 1] = 1
        rval[m - 1] = 2 * (y[m] - y[m - 1]) / h[m - 1]
    elif r == Boundary.CLAMPED:
        matrix[m-1][m-1] = 1
        rval[m-1] = 2 * (y[m] - y[m-1]) / h[m-1] - df(x[n-1])

    print(matrix)

    b = np.linalg.solve(matrix, rval)
    return b


def si(i, x0, x, y, b, c):
    return y[i] + b[i] * (x0 - x[i]) + c[i] * (x0 - x[i]) ** 2


def calculate(n, r):
    m = n - 1
    x = create_equidistant_nodes(A, B, n)
    y = f(x)
    h = get_h(x)
    b = spline(n, h, x, y, r)
    c = [(b[i + 1] - b[i]) / (2 * h[i]) for i in range(n - 2)]
    if r == Boundary.NATURAL:
        # w.brzegowy naturalny ... (chyba naturalny, taki, że f'(x_n) = 0)
        c.append(((y[m] - y[m - 1]) / h[m - 1] - b[m - 1]) / h[m - 1])
    elif r == Boundary.CLAMPED:
        c.append((df(x[n-1]) - b[-1]) / (2 * h[n-2]))

    x_axis = np.linspace(A, B, POINTS)
    y_axis = np.linspace(A, B, POINTS)

    ratio = POINTS / (n - 1)
    for i in range(POINTS):
        # TODO some issues may be here due to spline_index calculations
        spline_index = int(i // ratio)
        y_axis[i] = si(spline_index, x_axis[i], x, y, b, c)

    draw(x_axis, y_axis, x)


def draw(x, y, nodes):
    plt.plot(x, y)
    plt.plot(x, f(x))
    plt.scatter(nodes, f(nodes))
    plt.show()


def main():
    calculate(12, Boundary.CLAMPED)


if __name__ == "__main__":
    main()
