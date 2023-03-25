import numpy as np
import math
import matplotlib.pyplot as plt

POINTS = 500
A = -1 * np.pi
B = 2 * np.pi


def f(x, k=1, m=2):
    return np.exp(-k * np.sin(m * x)) + k * np.sin(m * x) - 1


def df(x, k=1, m=2):
    return k * m * np.cos(m * x) * (1 - np.exp(-1 * k * np.sin(m * x)))


def estimate_error(x, y):
    return np.max(np.abs(f(x) - y))


def estimate_error2(x, y):
    return np.sqrt(np.sum((f(x) - y) ** 2)) / len(x)


def chebyshev_nodes(x0, x1, n):
    nodes = []
    for i in range(1, n + 1, 1):
        node = 1 / 2 * (x0 + x1) + 1 / 2 * (x1 - x0) * math.cos((2 * i - 1) * math.pi / (2 * n))
        nodes.append(node)
    return nodes


def equidistant_nodes(x0, x1, n):
    nodes = []
    for i in range(n):
        node = x0 + i * (x1 - x0) / (n - 1)
        nodes.append(node)
    return nodes


def hermite_interpolation(nodes, f, df, x):
    n = len(nodes)
    z = []
    for i in range(n):
        z.extend([nodes[i], nodes[i]])
    n2 = 2 * n
    matrix = np.zeros((n2, n2))
    for i in range(n2):
        for j in range(i + 1):
            if j == 0:
                matrix[i][j] = f(z[i])
            elif j == 1 and i % 2 == 1:
                matrix[i][j] = df(z[i])
            else:
                matrix[i][j] = matrix[i][j - 1] - matrix[i - 1][j - 1]
                matrix[i][j] /= (z[i] - z[i - j])

    result = 0
    multiplier = 1
    for i in range(n2):
        result += matrix[i][i] * multiplier
        multiplier *= (x - z[i])
    return result


def draw(points, values, interpolated_values, n):
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Interpolacja metodą Hermite'a dla n = " + str(n))

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Interpolacja metodą Hermite dla n = ' + str(n))

    plt.plot(points, values, 'b-', points, interpolated_values, 'r-',
             equidistant_nodes(A, B, n), [f(node) for node in equidistant_nodes(A, B, n)], 'g.')
    plt.legend(['f(x)', 'interpolated', 'nodes'])
    plt.show()


def main():
    points = [A + i * (B - A) / POINTS for i in range(POINTS)]
    values = [f(point) for point in points]
    n = 20

    interpolated_values_regular = [hermite_interpolation(equidistant_nodes(A, B, n), f, df, point) for point in points]
    interpolated_values_chebyshev = [hermite_interpolation(chebyshev_nodes(A, B, n), f, df, point) for point in points]

    print(len(values))

    draw(points, values, interpolated_values_regular, n)
    draw(points, values, interpolated_values_chebyshev, n)

    differences = [interpolated_values_regular[i] - values[i] for i in range(POINTS)]
    print(n, end=' ')
    print(np.linalg.norm(differences))


if __name__ == "__main__":
    main()
