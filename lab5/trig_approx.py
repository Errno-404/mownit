import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

IMG_PATH = "./img"
POINTS = 1000

def func(x, k=3, m=3):
    return np.exp(-k * np.sin(m * x)) + k * np.sin(m * x) - 1


def draw(x, y, left, right, n, m, f):
    plt.scatter(x, y, label="data", color="red")
    xaxis = np.linspace(left, right, POINTS)
    plt.plot(xaxis, func(xaxis), label="F(x)", color="red")
    plt.plot(xaxis, f(xaxis), label="f(x)", color="blue")
    plt.title(f"Aproksymacja wielomianami trygonometrycznymi dla n = {n} i  m = {m}")
    plt.xlim(left, right)
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

    def compute_A_and_B(self):
        for i in range(self.n):
            ai = sum(self.Y[j] * np.cos(i * self.X[j]) for j in range(self.n))
            bi = sum(self.Y[j] * np.sin(i * self.X[j]) for j in range(self.n))
            self.A[i] = 2 * ai / self.n
            self.B[i] = 2 * bi / self.n

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
        return points


def calculate(left, right, n, m):
    X = np.linspace(left, right, n)
    Y = func(X)
    trigonometric_approximation = TrigonometricApproximation(X, Y, n, m, left, right)
    draw(X, Y, left, right, n, m, trigonometric_approximation.approximate)


calculate(-np.pi, 2 * np.pi, 100, 23)