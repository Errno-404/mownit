import numpy as np
import matplotlib.pyplot as plt


def fun(x: np.ndarray) -> np.ndarray:
    k = 3
    m = 3
    return np.exp(-k * np.sin(m * x)) + k * np.sin(m * x) - 1


def get_val(coeff, xi, x):
    return sum(elem * (x - xi) ** i for i, elem in enumerate(coeff))


def spline2(x_points, y_points, xs, boundary_cond):
    size = len(x_points) - 1
    matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i == j:
                matrix[i][j] = 1
            if j == i - 1:
                matrix[i][j] = 1

    g = np.zeros(size)
    h = []
    for i in range(size):
        h.append(x_points[i + 1] - x_points[i])
        g[i] = 2 / h[i] * (y_points[i + 1] - y_points[i])

    b = np.linalg.solve(matrix, g)

    if boundary_cond == 'natural':
        b = [0] + list(b)
    elif boundary_cond == 'not-a-knot':
        b = [b[-1]] + list(b)

    a = []
    c = []
    for i in range(size):
        a.append((b[i + 1] - b[i]) / (2 * h[i]))
        c.append(y_points[i])

    if boundary_cond == 'not-a-knot':
        b[0] = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
        a[0] = 0

    nr_fun = 0
    ys = []
    for i in range(len(xs)):
        while x_points[nr_fun + 1] < xs[i] < x_points[-1]:
            nr_fun += 1
        ys.append(get_val([c[nr_fun], b[nr_fun], a[nr_fun]], x_points[nr_fun], xs[i]))

    return ys


def draw(xs, fun_xs, x_points, y_points, ys_natural, ys_nak):
    plt.plot(xs, fun_xs)
    plt.plot(x_points, y_points, 'o', label='Data Points')
    plt.plot(xs, ys_natural, label='Natural')
    plt.plot(xs, ys_nak, label='Not-a-Knot')
    plt.legend()
    plt.show()


A = -np.pi
B = 2 * np.pi
POINTS = 500


def create_equidistant_points(n):
    return np.linspace(A, B, n)


def chebyshev_nodes(n):
    a = A
    b = B
    k = np.arange(1, n + 1)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n))
    return np.sort(nodes)


def calculate(n, nodes_type):
    if nodes_type == "regular":
        nodes_x = create_equidistant_points(n)
    else:
        nodes_x = chebyshev_nodes(n)
    nodes_y = fun(nodes_x)
    x_axis = np.linspace(A, B, POINTS)
    y_axis_natural = spline2(nodes_x, nodes_y, x_axis, 'natural')
    y_axis_knot = spline2(nodes_x, nodes_y, x_axis, 'not-a-knot')
    draw(x_axis, fun(x_axis), nodes_x, nodes_y, y_axis_natural, y_axis_knot)


# calculate(20, 'regular')


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
                        calculate(n, node_type)
                    except ValueError:
                        print("Invalid input. Please enter an integer.")


if __name__ == "__main__":
    main()
