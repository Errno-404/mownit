import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import csv

IMG_PATH = "./img"
ERROR_PATH = "./data"
A = -np.pi
B = 2 * np.pi
POINTS = 1000


def func(x, k=3, m=3):
    return np.exp(-k * np.sin(m * x)) + k * np.sin(m * x) - 1


def scale_to_2pi(X):
    x_prim = deepcopy(X)
    range_length = B - A
    x_prim = (x_prim - A) / range_length * 2 * np.pi - np.pi
    return x_prim


def scale_point_to_2pi(x):
    range_length = B - A
    return (x - A) / range_length * 2 * np.pi - np.pi


def compute_A_and_B(X, Y, n):
    left = 2 * np.array([sum(Y[j] * np.cos(i * X[j]) for j in range(n)) / n for i in range(n)])
    right = 2 * np.array([sum(Y[j] * np.sin(i * X[j]) for j in range(n)) / n for i in range(n)])
    return left, right


def approximate(X, a, b, m):
    points = []
    for x in X:
        x = deepcopy(x)
        x = scale_point_to_2pi(x)
        approximated_x = 1 / 2 * a[0] + sum(a[j] * np.cos(j * x) + b[j] * np.sin(j * x)
                                            for j in range(1, m + 1))
        points.append(approximated_x)
    return points


def calculate(n, m):
    x = np.linspace(A, B, n)
    y = func(x)

    a, b = compute_A_and_B(scale_to_2pi(x), y, n)
    xaxis = np.linspace(A, B, POINTS)
    yaxis = approximate(xaxis, a, b, m)

    draw(x, y, n, m, xaxis, yaxis)
    return estimate_error2(func, xaxis, yaxis)


def estimate_error2(f, x, y):
    return np.sqrt(np.sum((f(x) - y) ** 2)) / len(x)


def draw(x, y, n, m, xaxis, yaxis):
    plt.scatter(x, y, label="data", color="red")
    plt.plot(xaxis, func(xaxis), label="F(x)", color="red")
    plt.plot(xaxis, yaxis, label="f(x)", color="blue")
    plt.title(f"Aproksymacja wielomianami trygonometrycznymi dla n = {n} i  m = {m}")
    plt.xlim(A, B)
    plt.xticks([-3 * np.pi / 2, -np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi, 5 * np.pi / 2],
               ['-3π/2', '-π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π', '5π/2'])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc='best')
    plt.savefig(IMG_PATH + f"/img_{n}_{m}")
    plt.show()


def main():
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)

    if not os.path.exists(ERROR_PATH):
        os.makedirs(ERROR_PATH)

    if not os.path.isfile(ERROR_PATH + "/errors.csv"):
        with open(ERROR_PATH + "/errors.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["m", "n", "error"])

    csv_lines = []

    for n in range(5, 100, 5):
        for m in range(1, int((n - 1) / 2) + 1):
            row = [m, n, calculate(n, m)]
            csv_lines.append(row)

    for m in range(5, 100, 5):
        for n in range(2 * m + 1, 100, 5):
            row = [m, n, calculate(n, m)]
            csv_lines.append(row)

    with open(ERROR_PATH + "/errors.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        for row in csv_lines:
            writer.writerow(row)


if __name__ == "__main__":
    main()
