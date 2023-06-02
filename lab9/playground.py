import numpy as np
from jacoby_test import jacobi


def create_A_matrix(n, k=9, m=2.5):
    a = np.array([[0.0 for _ in range(n)] for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if i == j:
                a[i][j] = k
            else:
                a[i][j] = 1 / (abs(i - j) + m)
    return a


def calculate_d(A, x):
    return np.dot(A, x)


def generate_permutation_vector(n, seed=42, left=-1., right=1.):
    np.random.seed(seed)
    x = np.random.choice([left, right], size=n)
    return x


def euclidean_norm(x, x_sol):
    diff = x - x_sol
    norm = np.linalg.norm(diff)
    return norm


def max_norm(x, x_sol):
    diff = np.abs(x - x_sol)
    norm = np.linalg.norm(diff, ord=np.inf)
    return norm


if __name__ == "__main__":
    a = create_A_matrix(4)
    x = generate_permutation_vector(4)
    d = calculate_d(a, x)

    print(jacobi(a, d, 25))
    print(max_norm(x, jacobi(a, d, 25)))
