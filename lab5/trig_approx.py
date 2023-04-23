import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy

IMG_PATH = "./img"
A = -np.pi
B = 2 * np.pi
POINTS = 1000


def func(x, k=3, m=3):
    return np.exp(-k * np.sin(m * x)) + k * np.sin(m * x) - 1


def draw(x, y, n, m, xaxis, yaxis):
    plt.scatter(x, y, label="data", color="red")
    plt.plot(xaxis, func(xaxis), label="F(x)", color="red")
    plt.plot(xaxis, yaxis, label="f(x)", color="blue")
    plt.title(f"Aproksymacja wielomianami trygonometrycznymi dla n = {n} i  m = {m}")
    plt.xlim(A, B)
    xticks = np.arange(-3 * np.pi / 2, 5 * np.pi / 2 + 1e-9, np.pi / 2)  # dodajemy 1e-9, aby dodać 5π/2
    xticklabels = ['-3π/2', '-π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π', '5π/2']
    plt.xticks(xticks, xticklabels)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc='best')
    plt.savefig(IMG_PATH + f"/img_{n}_{m}")
    plt.show()


def scale_to_2pi(X):
    x_prim = deepcopy(X)
    n = len(x_prim)
    range_length = B - A
    for i in range(n):
        x_prim[i] = x_prim[i] / range_length
        x_prim[i] = x_prim[i] * 2 * np.pi
        x_prim[i] = x_prim[i] + -np.pi - (2 * np.pi * A / range_length)

    # print(x_prim)
    return x_prim

    # scale_to_2pi to prepare X
    # compute_A_and_B


def compute_A_and_B(X, Y, n):
    left = np.zeros(n)
    right = np.zeros(n)
    for i in range(n):
        ai = sum(Y[j] * np.cos(i * X[j]) for j in range(n))
        bi = sum(Y[j] * np.sin(i * X[j]) for j in range(n))
        left[i] = 2 * ai / n
        right[i] = 2 * bi / n
    return left, right


def scale_point_to_2pi(x):
    range_length = B - A
    x /= range_length
    x *= 2 * np.pi
    x += -np.pi - (2 * np.pi * A / range_length)
    return x


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

    scaled_x = scale_to_2pi(x)
    a, b = compute_A_and_B(scaled_x, y, n)

    xaxis = np.linspace(A, B, POINTS)
    yaxis = approximate(xaxis, a, b, m)

    draw(x, y, n, m, xaxis, yaxis)


def main():
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)

    calculate(50, 12)


if __name__ == "__main__":
    main()
