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


def draw(x, y, n, m, f):
    plt.scatter(x, y, label="data", color="red")
    xaxis = np.linspace(A, B, POINTS)
    plt.plot(xaxis, func(xaxis), label="F(x)", color="red")
    plt.plot(xaxis, f(xaxis), label="f(x)", color="blue")
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


class TrigonometricApproximation:
    def __init__(self, X, Y, n, m, start, stop):
        if m > np.floor((n - 1) / 2):
            raise Exception("m cannot be greater than floor of (n-1)/2")
        self.X = X
        self.Y = Y
        self.n = n
        self.m = m
        self.start = start
        self.stop = stop
        self.A = np.zeros(self.n)
        self.B = np.zeros(self.n)
        self.scale_to_2pi()
        self.compute_A_and_B()
        self.scale_from_2pi()

    def scale_to_2pi(self):
        range_length = self.stop - self.start
        for i in range(len(self.X)):
            self.X[i] /= range_length
            self.X[i] *= 2 * np.pi
            self.X[i] += -np.pi - (2 * np.pi * self.start / range_length)

        # print(self.X)

    # scale_to_2pi to prepare X
    # compute_A_and_B

    def compute_A_and_B(self):
        for i in range(self.n):
            ai = sum(self.Y[j] * np.cos(i * self.X[j]) for j in range(self.n))
            bi = sum(self.Y[j] * np.sin(i * self.X[j]) for j in range(self.n))
            self.A[i] = 2 * ai / self.n
            self.B[i] = 2 * bi / self.n

        print(self.A, self.B)

    def scale_from_2pi(self):
        range_length = self.stop - self.start
        for i in range(len(self.X)):
            self.X[i] -= -np.pi - (2 * np.pi * self.start / range_length)
            self.X[i] /= 2 * np.pi
            self.X[i] *= range_length

    def scale_point_to_2pi(self, x):
        range_length = self.stop - self.start
        x /= range_length
        x *= 2 * np.pi
        x += -np.pi - (2 * np.pi * self.start / range_length)
        return x

    def approximate(self, X):
        points = []
        for x in X:
            cp_x = deepcopy(x)
            cp_x = self.scale_point_to_2pi(cp_x)
            approximated_x = 1 / 2 * self.A[0] + sum(self.A[j] * np.cos(j * cp_x) + self.B[j] * np.sin(j * cp_x)
                                                     for j in range(1, self.m + 1))
            points.append(approximated_x)
        return self.A, self.B


def scale_to_2pi(X):
    x_prim = deepcopy(X)
    n = len(x_prim)
    range_length = B - A
    for i in range(n):
        x_prim[i] = X[i] / range_length
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


def approximate(X, A, B, m):
    points = []
    for x in X:
        cp_x = deepcopy(x)
        cp_x = scale_point_to_2pi(cp_x)
        approximated_x = 1 / 2 * A[0] + sum(A[j] * np.cos(j * cp_x) + B[j] * np.sin(j * cp_x)
                                            for j in range(1, m + 1))
        points.append(approximated_x)
    return points


def calculate(n, m):
    X = np.linspace(A, B, n)
    Y = func(X)



    scaled_x = scale_to_2pi(X)
    a, b = compute_A_and_B(scaled_x, Y, n)
    print(a, b)





    # yaxis = approximate(scaled_x, a, b, m)
    # xaxis = np.linspace(A, B, len(yaxis))






    trigonometric_approximation = TrigonometricApproximation(X, Y, n, m, A, B)
    a_p, b_p = trigonometric_approximation.approximate(X)
    for i in range(len(a)):
        if a[i] != a_p[i]:
            print(abs(a[i] - a_p[i]))
    # draw(X, Y, n, m, trigonometric_approximation.approximate)


def main():
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)

    calculate(100, 23)


if __name__ == "__main__":
    main()
