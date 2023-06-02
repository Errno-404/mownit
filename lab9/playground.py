import csv
import os.path

import numpy as np
from jacoby import jacoby, residual, increasing

MAX_ITER = 1000


def create_A_matrix(n, k=9, m=2.5):
    a = np.array([[0.0 for _ in range(n)] for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if i == j:
                a[i][j] = k
            else:
                a[i][j] = 1 / (abs(i - j) + m)
    return a


def calculate_d(a, x):
    return np.dot(a, x)


def generate_permutation_vector(n, seed=42, choose_from=None):
    if choose_from is None:
        choose_from = [-1., 1.]

    np.random.seed(seed)
    x = np.random.choice(choose_from, size=n)
    return x


def max_norm(x, x_sol):
    diff = np.abs(x - x_sol)
    norm = np.linalg.norm(diff, ord=np.inf)
    return norm


def main():
    ns = list(range(3, 21))
    epss = [1e-9, 1e-12, 1e-15]
    if not os.path.isfile("results_1.csv"):
        csv_header = ["eps", "norma - warunek przyrostowy", "norma - warunek rezydualny",
                      "iteracje - warunek przyrostowy", "iteracje - warunek rezydualny"]
        with open("results_1.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)

    if not os.path.isfile("results_2.csv"):
        csv_header = ["eps", "norma - warunek przyrostowy", "norma - warunek rezydualny",
                      "iteracje - warunek przyrostowy", "iteracje - warunek rezydualny"]
        with open("results_2.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)
    csv_line_1 = []
    csv_line_2 = []
    for eps in epss:

        for n in ns:
            # preparation to have A and B matrices to solve Ax' = B for x'
            a = create_A_matrix(n)
            x = generate_permutation_vector(n)
            d = calculate_d(a, x)

            # create 2 variants of starting vector
            x1 = np.zeros(n)
            x2 = generate_permutation_vector(n, choose_from=np.arange(100, 201))  # random from [100, 200]

            sol_inc_1, iter_inc_1 = jacoby(a, d, x1, increasing, MAX_ITER, eps)
            sol_res_1, iter_res_1 = jacoby(a, d, x1, residual, MAX_ITER, eps)

            sol_inc_2, iter_inc_2 = jacoby(a, d, x2, increasing, MAX_ITER, eps)
            sol_res_2, iter_res_2 = jacoby(a, d, x2, residual, MAX_ITER, eps)

            row = [eps, max_norm(x, sol_inc_1), max_norm(x, sol_res_1), iter_inc_1, iter_res_1]
            csv_line_1.append(row)

            row = [eps, max_norm(x, sol_inc_2), max_norm(x, sol_res_2), iter_inc_2, iter_res_2]
            csv_line_2.append(row)
            print(row)


    with open("results_1.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_line_1)

    with open("results_2.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_line_2)


if __name__ == "__main__":
    main()
