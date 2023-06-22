import numpy as np
import time


def create_tri_diagonal_matrix(n, m, k):
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)

    for i in range(1, n):
        a[i] = k / (i + 1 + m + 1)

    for i in range(n):
        b[i] = k

    for i in range(n - 1):
        c[i] = 1 / (i + 1 + m)

    return a, b, c


def solve_tridiagonal_system(a, b, c, d):
    n = len(d)
    c_star = np.zeros(n)
    d_star = np.zeros(n)
    x = np.zeros(n)

    start = time.time()
    # Step 1: Forward elimination
    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]

    for i in range(1, n - 1):
        c_star[i] = c[i] / (b[i] - a[i] * c_star[i - 1])

    for i in range(1, n):
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / (b[i] - a[i] * c_star[i - 1])

    # Step 2: Back substitution
    x[n - 1] = d_star[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]
    res = time.time() - start
    return x, res


def calculate_B(a, b, c, x, n):
    B = np.zeros(n)

    B[0] = b[0] * x[0] + c[1] * x[1]
    for i in range(1, n - 1):
        B[i] = a[i - 1] * x[i - 1] + b[i] * x[i] + c[i + 1] * x[i + 1]
    B[n - 1] = a[n - 2] * x[n - 2] + b[n - 1] * x[n - 1]

    return B


def generate_permutation_vector(n):
    np.random.seed(42)  # Ustalamy ziarno generatora liczb pseudolosowych
    x = np.random.choice([-1., 1.], size=n)
    return x


def euclidean_norm(x_dane, x_obliczone):
    x_dane = np.array(x_dane, dtype=np.float64)
    x_obliczone = np.array(x_obliczone, dtype=np.float64)
    diff = x_dane - x_obliczone
    norm = np.linalg.norm(diff)
    return norm


def max_norm(x_dane, x_obliczone):
    x_dane = np.array(x_dane, dtype=np.float64)
    x_obliczone = np.array(x_obliczone, dtype=np.float64)
    diff = np.abs(x_dane - x_obliczone)
    norm = np.max(diff)
    return norm


def main(n, m, k):
    a, b, c = create_tri_diagonal_matrix(n, m, k)
    x = generate_permutation_vector(n)
    B = calculate_B(a, b, c, x, n)
    x_solved, t = solve_tridiagonal_system(a, b, c, B)

    # print(euclidean_norm(x, x_solved))
    print(max_norm(x, x_solved), t)


if __name__ == "__main__":
    for i in range(10, 2001):
        main(i, 5, 4)
