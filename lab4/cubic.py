import numpy as np
from matplotlib import pyplot as plt
from enum import Enum
import os


class Boundary(Enum):
    NATURAL = 0
    CLAMPED = 1


A = -np.pi
B = 2 * np.pi
POINTS = 500


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


# returns value of i-th spline for x0
def get_si(i, x0, x, y, z, h):
    b = (y[i + 1] - y[i]) / h[i] - h[i] * (z[i + 1] + 2 * z[i])
    c = 3 * z[i]
    d = (z[i + 1] - z[i]) / h[i]
    return y[i] + b * (x0 - x[i]) + c * (x0 - x[i]) ** 2 + d * (x0 - x[i]) ** 3


def get_solutions(n, h, x, y, l_boundary=Boundary.NATURAL, r_boundary=Boundary.NATURAL):
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    rval = [0 for _ in range(n)]
    delta = [0 for _ in range(n - 1)]
    for i in range(n - 1):
        delta[i] = (y[i + 1] - y[i]) / h[i]

    for i in range(1, n - 1):
        matrix[i][i - 1] = h[i - 1]
        matrix[i][i] = 2 * (h[i] + h[i - 1])
        matrix[i][i + 1] = h[i]
        rval[i] = delta[i] - delta[i - 1]

    # left boundaries:
    if l_boundary == Boundary.NATURAL:
        matrix[0][0] = 1
    elif l_boundary == Boundary.CLAMPED:
        matrix[0][0] = 2
        matrix[0][1] = 1
        rval[0] = (delta[0] - df(x[0])) / h[0]

    # right boundaries
    if r_boundary == Boundary.NATURAL:
        matrix[n - 1][n - 1] = 1
    elif r_boundary == Boundary.CLAMPED:
        matrix[n - 1][n - 2] = h[n - 2]
        matrix[n - 1][n - 1] = 2
        rval[n - 1] = (delta[n - 2] - df(x[n - 1])) / h[n - 2]

    z = np.linalg.solve(matrix, rval)
    return z


def calculate(n, left, right):
    x = create_equidistant_nodes(A, B, n)
    y = f(x)
    h = get_h(x)
    z = get_solutions(n, h, x, y, left, right)

    x_axis = np.linspace(A, B, POINTS)
    y_axis = np.linspace(A, B, POINTS)

    ratio = POINTS / (n - 1)
    for i in range(POINTS):
        spline_index = int(i // ratio)
        y_axis[i] = get_si(spline_index, x_axis[i], x, y, z, h)

    draw(n, x_axis, y_axis, x, left, right)


def draw(n, x, y, nodes, left_boundary, right_boundary):
    directory = './img/cubic'
    if not os.path.exists(directory):
        os.makedirs(directory)

    left_boundary = 'clamped' if left_boundary == Boundary.CLAMPED else 'natural'
    right_boundary = 'clamped' if right_boundary == Boundary.CLAMPED else 'natural'

    filename = f'cubic_{n}_{left_boundary}_{right_boundary}.png'
    filepath = os.path.join(directory, filename)

    plt.plot(x, y)
    plt.scatter(nodes, f(nodes))
    plt.plot(x, f(x))
    plt.savefig(filepath)
    plt.show()


def main():
    while True:
        cmd = input('Type n (number of nodes) or exit')
        if cmd == 'exit':
            break
        elif cmd.isnumeric():
            n = int(cmd)

            left_boundary = Boundary.NATURAL
            right_boundary = Boundary.NATURAL
            correct = False
            exited = False
            while not correct:
                left_boundary = input('left boundary type: ')
                if left_boundary == 'c':
                    left_boundary = Boundary.CLAMPED
                    correct = True
                elif left_boundary == 'n':
                    left_boundary = Boundary.NATURAL
                    correct = True
                elif left_boundary == 'exit':
                    exited = True
                    break
                else:
                    print("Unknown command")

            correct = False
            while not correct and not exited:
                right_boundary = input('right boundary type: ')
                if right_boundary == 'c':
                    right_boundary = Boundary.CLAMPED
                    correct = True
                elif right_boundary == 'n':
                    right_boundary = Boundary.NATURAL
                    correct = True
                elif right_boundary == 'exit':
                    exited = True
                    break
                else:
                    print("Unknown command")

            if not exited:
                calculate(n, left_boundary, right_boundary)
        else:
            print("Unknown command")


if __name__ == "__main__":
    main()
