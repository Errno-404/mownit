import numpy as np
from matplotlib import pyplot as plt
from enum import Enum
import csv
import os

A = -np.pi
B = 2 * np.pi
POINTS = 500


class Boundary(Enum):
    NATURAL = 1
    CLAMPED = 2


def f(x, k=3, m=3):
    return np.exp(-k * np.sin(m * x)) + k * np.sin(m * x) - 1


def df(x, k=3, m=3):
    return k * m * np.cos(m * x) * (1 - np.exp(-k * np.sin(m * x)))


def create_equidistant_nodes(a, b, n):
    return np.linspace(a, b, n)


def get_h(x):
    n = len(x)
    return [x[i + 1] - x[i] for i in range(n - 1)]


def spline(n, h, x, y, r=Boundary.NATURAL):
    m = n - 1
    matrix = [[0 for _ in range(m)] for _ in range(m)]
    rval = [0 for _ in range(m)]
    for i in range(0, m - 1):
        matrix[i][i] = 1
        matrix[i][i + 1] = 1
        rval[i] = 2 * (y[i + 1] - y[i]) / h[i]

    # Boundary conditions
    if r == Boundary.NATURAL:
        matrix[m - 1][m - 1] = 1
        rval[m - 1] = 2 * (y[m] - y[m - 1]) / h[m - 1]
    elif r == Boundary.CLAMPED:
        matrix[m - 1][m - 1] = 1
        rval[m - 1] = 2 * (y[m] - y[m - 1]) / h[m - 1] - df(x[n - 1])

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

    # Boundary conditions
    if r == Boundary.NATURAL:
        c.append(((y[m] - y[m - 1]) / h[m - 1] - b[m - 1]) / h[m - 1])
    elif r == Boundary.CLAMPED:
        c.append((df(x[n - 1]) - b[-1]) / (2 * h[n - 2]))

    x_axis = np.linspace(A, B, POINTS)
    y_axis = np.linspace(A, B, POINTS)

    ratio = POINTS / (n - 1)
    for i in range(POINTS):
        spline_index = int(i // ratio)
        y_axis[i] = si(spline_index, x_axis[i], x, y, b, c)

    draw(n, x_axis, y_axis, x, r)
    return estimate_error(x_axis, y_axis), estimate_error2(x_axis, y_axis)


def estimate_error(x, y):
    return np.max(np.abs(f(x) - y))


def estimate_error2(x, y):
    return np.sqrt(np.sum((f(x) - y) ** 2)) / len(x)


def draw(n, x, y, nodes, right_boundary):
    output_dir = './img/quadratic'
    os.makedirs(output_dir, exist_ok=True)

    right_boundary_str = 'clamped' if right_boundary == Boundary.CLAMPED else 'natural'

    filename = f'quadratic_{n}_{right_boundary_str}.png'
    filepath = os.path.join(output_dir, filename)

    plt.plot(x, y, label='w(x) - funkcja sklejana', color='blue')
    plt.scatter(nodes, f(nodes), label='węzeł')
    plt.plot(x, f(x), label='f(x) - funkcja interpolowana', color='red')

    title = f'Interpolacja funkcjami sklejanymi stopnia 2, n = {n}\nWarunek brzegowy: {right_boundary_str}'
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
    # Create the errors directory if it does not exist
    if not os.path.exists('./errors/quadratic'):
        os.makedirs('./errors/quadratic')

    # Open the CSV file for writing the headers
    file_path = './errors/quadratic/errors.csv'
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['n', 'boundary_type', 'err_1', 'err_2'])

    while True:
        cmd = input('Type n (number of nodes) or exit: ')
        if cmd == 'exit':
            break
        elif cmd.isnumeric():
            n = int(cmd)
            right_boundary = None

            # get right boundary type from user
            while right_boundary not in [Boundary.CLAMPED, Boundary.NATURAL]:
                right_boundary = input('Right boundary type (c/n/exit): ').strip()
                if right_boundary == 'exit':
                    break
                elif right_boundary == 'c':
                    right_boundary = Boundary.CLAMPED
                elif right_boundary == 'n':
                    right_boundary = Boundary.NATURAL
                else:
                    print("Unknown command")

            if right_boundary is not None:
                err_1, err_2 = calculate(n, right_boundary)

                # save errors to the CSV file
                file_path = './errors/quadratic/errors.csv'
                with open(file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([n, right_boundary.name, err_1, err_2])
        else:
            print("Unknown command")


if __name__ == "__main__":
    main()
