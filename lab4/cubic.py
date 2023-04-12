import numpy as np
from matplotlib import pyplot as plt
from enum import Enum
import os
import csv


class Boundary(Enum):
    NATURAL = 0
    CLAMPED = 1


A = -np.pi
B = 2 * np.pi
POINTS = 1000


def f(x, k=3, m=3):
    return np.exp(-k * np.sin(m * x)) + k * np.sin(m * x) - 1


def df(x, k=3, m=3):
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
    rval = np.zeros(n)
    delta = np.zeros(n - 1)
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
        matrix[0][0] = 2 * h[0]
        matrix[0][1] = h[0]
        rval[0] = delta[0] - df(x[n - 1])

    # right boundaries
    if r_boundary == Boundary.NATURAL:
        matrix[n - 1][n - 1] = 1
    elif r_boundary == Boundary.CLAMPED:
        matrix[n - 1][n - 2] = h[n - 2]
        matrix[n - 1][n - 1] = 2 * h[n - 2]
        rval[n - 1] = df(x[n - 1]) - delta[n - 2]

    z = np.linalg.solve(matrix, rval)
    return z


def calculate(n, left_boundary, right_boundary):
    x = create_equidistant_nodes(A, B, n)
    y = f(x)
    h = get_h(x)
    z = get_solutions(n, h, x, y, left_boundary, right_boundary)

    x_axis = np.linspace(A, B, POINTS)
    y_axis = np.empty_like(x_axis)

    ratio = POINTS / (n - 1)
    for i in range(POINTS):
        spline_index = int(i // ratio)
        # print(spline_index)
        y_axis[i] = get_si(spline_index, x_axis[i], x, y, z, h)

    draw(n, x_axis, y_axis, x, left_boundary, right_boundary)
    return estimate_error(x_axis, y_axis), estimate_error2(x_axis, y_axis)


def estimate_error(x, y):
    return np.max(np.abs(f(x) - y))


def estimate_error2(x, y):
    return np.sqrt(np.sum((f(x) - y) ** 2)) / len(x)


def draw(n, x, y, nodes, left_boundary, right_boundary):
    output_dir = './img/cubic'
    os.makedirs(output_dir, exist_ok=True)

    left_boundary_str = 'clamped' if left_boundary == Boundary.CLAMPED else 'natural'
    right_boundary_str = 'clamped' if right_boundary == Boundary.CLAMPED else 'natural'

    filename = f'cubic_{n}_{left_boundary_str}_{right_boundary_str}.png'
    filepath = os.path.join(output_dir, filename)

    plt.plot(x, y, label='w(x) - funkcja sklejana', color='blue')
    plt.scatter(nodes, f(nodes), label='węzeł')
    plt.plot(x, f(x), label='f(x) - funkcja interpolowana', color='red')

    title = f'Interpolacja funkcjami sklejanymi stopnia 3, n = {n}\nWarunki brzegowe: {left_boundary_str},' \
            f' {right_boundary_str}'
    plt.title(title)
    plt.xlim(A, B)
    xticks = np.arange(-3 * np.pi / 2, 5 * np.pi / 2 + 1e-9, np.pi / 2)
    xticklabels = ['-3π/2', '-π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π', '5π/2']
    plt.xticks(xticks, xticklabels)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')

    plt.savefig(filepath)
    plt.show()


def main():
    dir_path = './errors/cubic'
    file_path = os.path.join(dir_path, 'error.csv')
    header = ['n', 'left_boundary', 'right_boundary', 'err_1', 'err_2']

    # create directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)

        while True:
            cmd = input('Type n (number of nodes) or exit: ')
            if cmd == 'exit':
                break
            elif cmd.isnumeric():
                n = int(cmd)
                left_boundary = None
                right_boundary = None

                # get left boundary type from user
                while left_boundary not in [Boundary.CLAMPED, Boundary.NATURAL]:
                    left_boundary = input('Left boundary type (c/n/exit): ').strip()
                    if left_boundary == 'exit':
                        break
                    elif left_boundary == 'c':
                        left_boundary = Boundary.CLAMPED
                    elif left_boundary == 'n':
                        left_boundary = Boundary.NATURAL
                    else:
                        print("Unknown command")

                # get right boundary type from user
                while right_boundary not in [Boundary.CLAMPED, Boundary.NATURAL] and left_boundary is not None:
                    right_boundary = input('Right boundary type (c/n/exit): ').strip()
                    if right_boundary == 'exit':
                        break
                    elif right_boundary == 'c':
                        right_boundary = Boundary.CLAMPED
                    elif right_boundary == 'n':
                        right_boundary = Boundary.NATURAL
                    else:
                        print("Unknown command")

                if left_boundary is not None and right_boundary is not None:
                    err_1, err_2 = calculate(n, left_boundary, right_boundary)
                    writer.writerow([n, left_boundary, right_boundary, err_1, err_2])


if __name__ == "__main__":
    main()
