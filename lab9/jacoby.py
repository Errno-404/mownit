import numpy as np


def jacoby(a, b, x0, condition, max_iterations, eps):
    n = len(a)
    x = np.copy(x0)
    x_old = np.copy(x0)
    iteration = 0

    while iteration < max_iterations and not condition(a, x, b, eps, x_old, iteration):
        for i in range(n):
            suma = 0
            for j in range(n):
                if j != i:
                    suma += a[i][j] * x[j]
            x_old[i] = x[i]
            x[i] = (1 / a[i][i]) * (b[i] - suma)
        iteration += 1

    return x, iteration


def increasing(A, x, b, eps, x_old, iteration):
    if np.array_equal(x, x_old) and iteration == 0:
        return False
    return np.linalg.norm(x - x_old) < eps


def residual(A, x, b, eps, x_old, iteration):
    return np.linalg.norm(np.dot(A, x) - b) < eps


if __name__ == "__main__":
    A = np.array([[2.0, 1.0], [5.0, 7.0]])
    b = np.array([11.0, 13.0])
    guess = np.array([1.0, 1.0])

    print(jacoby(A, b, guess, residual, 100, 0.00000001))
    print(jacoby(A, b, guess, increasing, 100, 0.00000001))
