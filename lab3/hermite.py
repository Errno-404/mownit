import numpy as np
import matplotlib.pyplot as plt


def fun(x: np.ndarray) -> np.ndarray:
    k = 3
    m = 3
    return np.exp(-k * np.sin(m * x)) + k * np.sin(m * x) - 1


def dfun(x: np.ndarray) -> np.ndarray:
    k = 3
    m = 3
    return k * m * np.cos(m * x) * (1 - np.exp(-1 * k * np.sin(m * x)))


def estimate_error(x: np.ndarray, y: np.ndarray, f) -> float:
    return np.max(np.abs(f(x) - y))


def estimate_rms_error(x: np.ndarray, y: np.ndarray, f) -> float:
    return np.sqrt(np.mean((f(x) - y) ** 2))


def create_chebyshev_nodes(x0: float, x1: float, n: int) -> np.ndarray:
    nodes = np.zeros(n)
    for i in range(1, n + 1, 1):
        node = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * np.cos((2 * i - 1) * np.pi / (2 * n))
        nodes[i - 1] = node
    return nodes


def create_regular_nodes(x0: float, x1: float, n: int) -> np.ndarray:
    return np.linspace(x0, x1, n)


POINTS = 500
A = -1 * np.pi
B = 2 * np.pi


def hermite(nodes, f, df, x):
    z = np.repeat(nodes, 2)  # use numpy.repeat instead of a list comprehension
    n2 = 2 * len(nodes)
    matrix = np.zeros((n2, n2))

    for i in range(n2):
        for j in range(i + 1):
            if j == 0:
                matrix[i][j] = f(z[i])
            elif j == 1 and i % 2 == 1:
                matrix[i][j] = df(z[i])
            else:
                matrix[i][j] = (matrix[i][j - 1] - matrix[i - 1][j - 1]) / (z[i] - z[i - j])

    result, multiplier = 0, 1
    for i, matrix_element in enumerate(np.diag(matrix)):
        result += matrix_element * multiplier
        multiplier *= (x - z[i])

    return result


def polynom(nodes, f, df):
    x_axis = np.linspace(A, B, POINTS)
    y_axis = np.array([hermite(nodes, f, df, x) for x in x_axis])  # use numpy.array instead of a for loop

    return x_axis, y_axis


def draw(x_axis, y_axis, nodes, nodes_type):
    plt.plot(x_axis, fun(x_axis), color='red', linestyle='-', label='f(x) - funkcja interpolowana')
    plt.plot(x_axis, y_axis, color='blue', linestyle='-', label='w(x) - wielomain interpolujący')
    plt.scatter(nodes, fun(nodes), label='węzeł')

    if nodes_type == "chebyshev":
        plt.title("Interpolacja Hermite'a dla węzłów Czebyszewa")

    else:
        plt.title("Interpolacja Hermite'a dla węzłów równoodległych")

    plt.xlim(A, B)
    xticks = np.arange(-3 * np.pi / 2, 5 * np.pi / 2 + 1e-9, np.pi / 2)  # dodajemy 1e-9, aby dodać 5π/2
    xticklabels = ['-3π/2', '-π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π', '5π/2']
    plt.xticks(xticks, xticklabels)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')

    plt.savefig("test.png")
    plt.show()


def calculate(n, nodes_type):
    if nodes_type == "chebyshev":
        nodes = create_chebyshev_nodes(A, B, n)
    else:
        nodes = create_regular_nodes(A, B, n)

    x_axis, y_axis = polynom(nodes, fun, dfun)
    err1 = estimate_error(x_axis, y_axis, fun)
    err2 = estimate_rms_error(x_axis, y_axis, fun)
    draw(x_axis, y_axis, nodes, nodes_type)

    return err1, err2


def main():
    n = 11
    err1, err2 = calculate(n, "regular")
    print(f"Error 1: {err1}, Error 2: {err2}")


if __name__ == "__main__":
    main()
