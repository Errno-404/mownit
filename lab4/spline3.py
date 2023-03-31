# Wektor l zawiera wartości diagonalne macierzy trójdiagonalnej, która pojawia się w procesie obliczania parametrów funkcji sklejanej. Wartości na diagonali są wyliczane zgodnie z następującym wzorem:


# l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
# gdzie h to różnice między kolejnymi wartościami wektora x, a mu to wektor zawierający wartości elementów poddiagonalnych macierzy.
#
# Wektor mu zawiera wartości elementów poddiagonalnych macierzy trójdiagonalnej, które również są wyliczane w trakcie obliczania parametrów funkcji sklejanej zgodnie z wzorem:

# mu[i] = h[i] / l[i]
# Wektor z jest wynikiem rozwiązania układu równań uzyskanego po zastosowaniu metody eliminacji Gaussa z częściowym wyborem elementu głównego. Jest on wykorzystywany w dalszej części obliczeń do wyznaczenia parametrów funkcji sklejanej. Wektor z jest obliczany w pętli for w funkcji calculate_cubic_spline(x, y) i jego wartości są wyliczane zgodnie z poniższym wzorem:

# z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]
# gdzie alpha to wektor zawierający wartości pierwszych pochodnych funkcji sklejanej w punktach danych.

import numpy as np
import matplotlib.pyplot as plt


def fun(x: np.ndarray) -> np.ndarray:
    k = 3
    m = 3
    return np.exp(-k * np.sin(m * x)) + k * np.sin(m * x) - 1


import numpy as np

def calculate_cubic_spline(x, y, bc='natural'):
    n = len(x)
    h = np.diff(x)
    alpha = np.zeros(n)
    for i in range(1, n - 1):
        alpha[i] = 3 * (y[i + 1] - y[i]) / h[i] - 3 * (y[i] - y[i - 1]) / h[i - 1]
    if bc == 'natural':
        l = np.ones(n - 1)
        mu = np.zeros(n - 1)
        z = np.zeros(n)
        z[0] = z[n - 1] = 0
    elif bc == 'not-a-knot':
        l = np.zeros(n)
        mu = np.zeros(n)
        z = np.zeros(n + 1)
        l[0] = h[1] + h[0]
        mu[0] = -h[1] / l[0]
        z[0] = -alpha[1] + alpha[0]
        l[n - 1] = h[n - 2] + h[n - 3]
        mu[n - 2] = -h[n - 3] / l[n - 1]
        z[n] = alpha[n - 2] - alpha[n - 1]
    else:
        raise ValueError('Invalid boundary condition specified')

    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    if bc == 'not-a-knot':
        z[1] -= mu[0] * z[0]
        z[n - 1] -= mu[n - 2] * z[n]

    c = np.zeros(n)
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    return b, c, d


def evaluate_spline(x_interp, x, y, b, c, d):
    y_interp = np.zeros_like(x_interp)
    for i in range(len(x_interp)):
        j = np.searchsorted(x, x_interp[i]) - 1
        if j < 0:
            j = 0
        elif j > len(x) - 2:
            j = len(x) - 2
        dx = x_interp[i] - x[j]
        y_interp[i] = y[j] + b[j] * dx + c[j] * dx ** 2 + d[j] * dx ** 3
    return y_interp



# def calculate_cubic_spline(x, y):
#     n = len(x)
#     h = np.diff(x)
#     alpha = np.zeros(n)
#     for i in range(1, n - 1):
#         alpha[i] = 3 * (y[i + 1] - y[i]) / h[i] - 3 * (y[i] - y[i - 1]) / h[i - 1]
#     l = np.zeros(n)
#     mu = np.zeros(n)
#     z = np.zeros(n)
#     l[0] = 1
#     mu[0] = 0
#     z[0] = 0
#     for i in range(1, n - 1):
#         l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
#         mu[i] = h[i] / l[i]
#         z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
#     l[n - 1] = 1
#     z[n - 1] = 0
#     c = np.zeros(n)
#     b = np.zeros(n - 1)
#     d = np.zeros(n - 1)
#     for j in range(n - 2, -1, -1):
#         c[j] = z[j] - mu[j] * c[j + 1]
#         b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
#         d[j] = (c[j + 1] - c[j]) / (3 * h[j])
#     return b, c, d
#
#
# def evaluate_spline(x_interp, x, y, b, c, d):
#     y_interp = np.zeros_like(x_interp)
#     for i in range(len(x_interp)):
#         j = np.searchsorted(x, x_interp[i]) - 1
#         if j < 0:
#             j = 0
#         elif j > len(x) - 2:
#             j = len(x) - 2
#         dx = x_interp[i] - x[j]
#         y_interp[i] = y[j] + b[j] * dx + c[j] * dx ** 2 + d[j] * dx ** 3
#     return y_interp


def create_chebyshev_nodes(x0: float, x1: float, n: int) -> np.ndarray:
    nodes = np.zeros(n)
    for i in range(1, n + 1, 1):
        node = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * np.cos((2 * i - 1) * np.pi / (2 * n))
        nodes[i - 1] = node
    return nodes


def create_regular_nodes(x0: float, x1: float, n: int) -> np.ndarray:
    return np.linspace(x0, x1, n)


def plot_spline(x, y, x_axis, y_axis_natural, y_axis_knot):
    plt.plot(x_axis, fun(x_axis), label='f(x)')
    plt.plot(x_axis, y_axis_natural, label='spline, natural')
    plt.plot(x_axis, y_axis_knot, label='spline, knot')

    plt.plot(x, y, 'o', label='data')
    plt.legend(loc='best')
    plt.show()


A = -np.pi
B = 2 * np.pi
POINTS = 500


def calculate_everything(nodes_type, n):
    if nodes_type == "regular":
        nodes = create_regular_nodes(A, B, n)
    else:
        nodes = create_chebyshev_nodes(A, B, n)
        nodes.sort()

    x_axis = np.linspace(A, B, POINTS)


    b_n, c_n, d_n = calculate_cubic_spline(nodes, fun(nodes), 'natural')
    b_k, c_k, d_k = calculate_cubic_spline(nodes, fun(nodes), 'not-a-knot')


    y_axis_natural = evaluate_spline(x_axis, nodes, fun(nodes), b_n, c_n, d_n)
    y_axis_knot = evaluate_spline(x_axis, nodes, fun(nodes), b_k, c_k, d_k)


    plot_spline(nodes, fun(nodes), x_axis, y_axis_natural, y_axis_knot)


def main():
    node_type = None

    while True:
        # Ask user to choose between regular and Chebyshev nodes or exit
        node_type_input = input("Choose node type (regular, chebyshev or exit): ")
        if node_type_input == "exit":
            break
        elif node_type_input == "regular" or node_type_input == "chebyshev":
            node_type = node_type_input
        else:
            print("Invalid node type. Please choose 'regular', 'chebyshev' or 'exit'.")

        if node_type:
            # Ask user for the number of nodes or exit to choose another mode
            while True:
                user_input = input("Enter an integer, or exit to choose another mode: ")
                if user_input == "exit":
                    break
                else:
                    try:
                        n = int(user_input)
                        calculate_everything(node_type, n)
                    except ValueError:
                        print("Invalid input. Please enter an integer.")


if __name__ == "__main__":
    main()
