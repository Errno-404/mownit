from gauss import gauss
from thomas import thomas
import numpy as np
import csv
import os


def create_matrices_thomas(n, m=5, k=4):
    a = np.array([k / ((i + 1) + m + 1) for i in range(1, n)])
    b = np.array([k for _ in range(n)])
    c = np.array([1 / ((i + 1) + m) for i in range(n - 1)])

    return a, b, c


def create_matrix_gauss(n, m=5, k=4):
    a = np.array([[0.0 for _ in range(n)] for _ in range(n)])

    a[0][0] = k  # b
    a[0][1] = 1 / (1 + m)  # c

    a[n - 1][n - 2] = k / (n + m + 1)  # a
    a[n - 1][n - 1] = k  # b

    for i in range(1, n - 1):
        a[i][i - 1] = k / ((i + 1) + m + 1)  # a
        a[i][i] = k  # b
        a[i][i + 1] = 1 / ((i + 1) + m)  # c

    return a


def calculate_d_thomas(a, b, c, x):
    n = len(x)
    d = np.array([0.0 for _ in range(n)])

    for i in range(1, n - 1):
        d[i] = a[i - 1] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1]
    d[0] = b[0] * x[0] + c[0] * x[1]
    d[n - 1] = b[n - 1] * x[n - 1] + a[n - 2] * x[n - 2]

    return d


def calculate_d_gauss(A, x):
    return np.dot(A, x)


def generate_permutation_vector(n):
    np.random.seed(42)  # Ustalamy ziarno generatora liczb pseudolosowych
    x = np.random.choice([-1., 1.], size=n)
    return x


def euclidean_norm(x, x_sol):
    diff = x - x_sol
    norm = np.linalg.norm(diff)
    return norm


def max_norm(x, x_sol):
    diff = np.abs(x - x_sol)
    norm = np.linalg.norm(diff, ord=np.inf)
    return norm


def main():
    # np.set_printoptions(precision=15)

    ns = [i for i in range(3, 21)]
    ns = ns + [50, 100, 200, 500, 1000, 1500]

    to_csv = []

    if not os.path.isfile("results.csv"):
        csv_header = ["n", "T norm", "G norm", "T time", "G time", "T mem", "G mem"]
        with open("results.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)

    for n in ns:
        a, b, c = create_matrices_thomas(n)
        A = create_matrix_gauss(n)
        x = generate_permutation_vector(n)

        t_d = calculate_d_thomas(a, b, c, x)
        g_d = calculate_d_gauss(A, x)

        t_sol, t_space, t_time = thomas(a, b, c, t_d)
        g_sol, g_space, g_time = gauss(A, g_d)

        row = [n, max_norm(x, t_sol), max_norm(x, g_sol), t_time, g_time, t_space, g_space]
        to_csv.append(row)
        print(f"n: {n}")
        print(
            f"\tThomas:\t\tczas: {t_time:.3f} s, pamięć: {t_space} B, dokładność (norma euklidesowa): "
            f"{euclidean_norm(x, t_sol)}, dokładność (norma maksimum) : {max_norm(x, t_sol)}")
        print(
            f"\tGauss:\t\tczas: {g_time:.3f} s, pamięć: {g_space} B, dokładność (norma euklidesowa): "
            f"{euclidean_norm(x, g_sol)}, dokładność (norma maksimum) : {max_norm(x, t_sol)}")

    with open("results.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(to_csv)


if __name__ == "__main__":
    main()
