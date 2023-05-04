import numpy as np
from matplotlib import pyplot as plt
import os
import csv

IMG_PATH = "./img"
ERROR_PATH = "./data"

A = -np.pi
B = 2 * np.pi
POINTS = 1000


def fun(x, k=3, m=3):
    return np.exp(-k * np.sin(m * x)) + k * np.sin(m * x) - 1


def scale_intervals(from_a, from_b, to_c, to_d, x):
    return to_c + (to_d - to_c) / (from_b - from_a) * (x - from_a)


def get_a_j(j, f, x, n):
    return 2 / n * sum([f(x[i]) * np.cos(j * x[i]) for i in range(n)])


def get_b_j(j, f, x, n):
    return 2 / n * sum([f(x[i]) * np.sin(j * x[i]) for i in range(n)])


def approximate(x, a, b, m):
    return 1 / 2 * a[0] + sum([a[j] * np.cos(j * x) + b[j] * np.sin(j * x) for j in range(1, m + 1)])


def calculate(f, n, m):
    # trzeba potestować czym jest n a czym m
    x = np.linspace(A, B, n)
    y = f(x)

    x_scaled = scale_intervals(A, B, -np.pi, np.pi, x)

    # print(x_scaled)

    a = [get_a_j(j, f, x_scaled, n) for j in range(m + 1)]
    b = [get_b_j(j, f, x_scaled, n) for j in range(m + 1)]

    print(a, b)

    xaxis = np.linspace(A, B, POINTS)
    yaxis = approximate(xaxis, a, b, m)

    draw(f, x, y, n, m, xaxis, yaxis)
    return estimate_error2(f, xaxis, yaxis)


def estimate_error2(f, x, y):
    return np.sqrt(np.sum((f(x) - y) ** 2)) / len(x)


def draw(f, x, y, n, m, xaxis, yaxis):
    plt.scatter(x, y, label="data", color="red")
    plt.plot(xaxis, f(xaxis), label="F(x)", color="red")
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


if __name__ == "__main__":
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)

    if not os.path.exists(ERROR_PATH):
        os.makedirs(ERROR_PATH)

    if not os.path.isfile(ERROR_PATH + "/errors.csv"):
        with open(ERROR_PATH + "/errors.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["m", "n", "error"])

    csv_lines = []
