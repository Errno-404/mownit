import numpy as np
import csv
import os


def gauss_elimination_float32(A, B):
    n = len(A)
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)

    for k in range(n - 1):
        p = np.argmax(np.abs(A[k:, k])) + k
        A[[k, p]] = A[[p, k]]
        B[[k, p]] = B[[p, k]]

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            B[i] -= m * B[k]

    x = np.zeros(n, dtype=np.float32)
    for i in range(n - 1, -1, -1):
        s = np.dot(A[i, i + 1:], x[i + 1:])
        x[i] = (B[i] - s) / A[i, i]

    return x


def gauss_elimination_float64(A, B):
    n = len(A)
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)

    for k in range(n - 1):
        p = np.argmax(np.abs(A[k:, k])) + k
        A[[k, p]] = A[[p, k]]
        B[[k, p]] = B[[p, k]]

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            B[i] -= m * B[k]

    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = np.dot(A[i, i + 1:], x[i + 1:])
        x[i] = (B[i] - s) / A[i, i]

    return x


def create_matrix_A_float32(n):
    A = np.zeros((n, n), dtype=np.float32)
    A[0, :] = 1.0

    for i in range(1, n):
        for j in range(n):
            A[i, j] = 1.0 / np.float32(i + j + 1)

    return A


def create_matrix_A_float64(n):
    A = np.zeros((n, n), dtype=np.float64)
    A[0, :] = 1.0

    for i in range(1, n):
        for j in range(n):
            A[i, j] = 1.0 / np.float64(i + j + 1)

    return A


def create_matrix_A_2_float64(n):
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if j >= i:
                A[i][j] = 2 * (i + 1) / (j + 1)
            else:
                A[i][j] = A[j][i]
    return A


def create_matrix_A_2_float32(n):
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if j >= i:
                A[i][j] = 2 * (i + 1) / (j + 1)
            else:
                A[i][j] = A[j][i]
    return A


def calculate_B_float32(A, x):
    A = np.array(A, dtype=np.float32)
    x = np.array(x, dtype=np.float32)
    B = np.dot(A, x)
    return B


def calculate_B_float64(A, x):
    A = np.array(A, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    B = np.dot(A, x)
    return B


def generate_permutation_vector(n):
    np.random.seed(42)  # Ustalamy ziarno generatora liczb pseudolosowych
    x = np.random.choice([-1., 1.], size=n)
    return x


def euclidean_norm_float64(x_dane, x_obliczone):
    x_dane = np.array(x_dane, dtype=np.float64)
    x_obliczone = np.array(x_obliczone, dtype=np.float64)
    diff = x_dane - x_obliczone
    norm = np.linalg.norm(diff)
    return norm


def euclidean_norm_float32(x_dane, x_obliczone):
    x_dane = np.array(x_dane, dtype=np.float32)
    x_obliczone = np.array(x_obliczone, dtype=np.float32)
    diff = x_dane - x_obliczone
    norm = np.linalg.norm(diff)
    return norm


def max_norm_float64(x_dane, x_obliczone):
    x_dane = np.array(x_dane, dtype=np.float64)
    x_obliczone = np.array(x_obliczone, dtype=np.float64)
    diff = np.abs(x_dane - x_obliczone)
    norm = np.max(diff)
    return norm


def max_norm_float32(x_dane, x_obliczone):
    x_dane = np.array(x_dane, dtype=np.float32)
    x_obliczone = np.array(x_obliczone, dtype=np.float32)
    diff = np.abs(x_dane - x_obliczone)
    norm = np.max(diff)
    return norm


def zad1():
    res_64 = []
    res_32 = []
    for n in range(3, 21):
        x = generate_permutation_vector(n)
        print("========== Float64 ==========")
        a = create_matrix_A_float64(n)
        b = calculate_B_float64(a, x)
        x_64 = gauss_elimination_float64(a, b)
        e_norm_64 = euclidean_norm_float64(x, x_64)
        m_norm_64 = max_norm_float64(x, x_64)
        res_64.append([n, e_norm_64, m_norm_64])

        print(f"\t n = {n}")
        print("\t x_origin: ", x)
        print("\t x_solved: ", x_64)
        print("\t euclidean_norm: ", e_norm_64)
        print("\t maximum_norm: ", m_norm_64)
        print()

        print()
        print("========== Float32 ==========")
        a = create_matrix_A_float32(n)
        b = calculate_B_float32(a, x)
        x_32 = gauss_elimination_float32(a, b)
        e_norm_32 = euclidean_norm_float32(x, x_32)
        m_norm_32 = max_norm_float32(x, x_32)
        res_32.append([n, e_norm_32, m_norm_32])

        print(f"\t n = {n}")
        print("\t x_origin: ", x)
        print("\t x_solved: ", x_32)
        print("\t euclidean_norm: ", e_norm_32)
        print("\t maximum_norm: ", m_norm_32)
    return res_32, res_64


def zad2():
    res_64 = []
    res_32 = []
    matrix_sizes = [n for n in range(3, 21)]
    matrix_sizes += [50, 100, 200, 500, 1000, 1500, 2000] # 5000, 10_000
    for n in matrix_sizes:
        x = generate_permutation_vector(n)
        print("========== Float64 ==========")
        a = create_matrix_A_2_float64(n)
        b = calculate_B_float64(a, x)
        x_64 = gauss_elimination_float64(a, b)
        e_norm_64 = euclidean_norm_float64(x, x_64)
        m_norm_64 = max_norm_float64(x, x_64)
        res_64.append([n, e_norm_64, m_norm_64])

        print(f"\t n = {n}")
        print("\t x_origin: ", x)
        print("\t x_solved: ", x_64)
        print("\t euclidean_norm: ", e_norm_64)
        print("\t maximum_norm: ", m_norm_64)
        print()

        print("========== Float32 ==========")
        a = create_matrix_A_2_float32(n)
        b = calculate_B_float32(a, x)
        x_32 = gauss_elimination_float32(a, b)
        e_norm_32 = euclidean_norm_float32(x, x_32)
        m_norm_32 = max_norm_float32(x, x_32)
        res_32.append([n, e_norm_32, m_norm_32])

        print(f"\t n = {n}")
        print("\t x_origin: ", x)
        print("\t x_solved: ", x_32)
        print("\t euclidean_norm: ", e_norm_32)
        print("\t maximum_norm: ", m_norm_32)
    return res_32, res_64


if __name__ == "__main__":
    header = ["n", "euclidean_norm", "maximum_norm"]
    print(".................. Zadanie 1 ...............")
    res_32, res_64 = zad1()
    ex_1_32_exists = False
    ex_1_64_exists = False
    if os.path.isfile("ex_1_32.csv"):
        ex_1_32_exists = True
    if os.path.isfile("ex_1_64.csv"):
        ex_1_64_exists = True

    with open("ex_1_32.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        if not ex_1_32_exists:
            writer.writerow(header)
        writer.writerows(res_32)

    with open("ex_1_64.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        if not ex_1_64_exists:
            writer.writerow(header)
        writer.writerows(res_64)

    print("\n\n\n\n\n.................. Zadanie 2 ...............")
    res_32, res_64 = zad2()

    ex_2_32_exists = False
    ex_2_64_exists = False
    if os.path.isfile("ex_2_32.csv"):
        ex_2_32_exists = True
    if os.path.isfile("ex_2_64.csv"):
        ex_2_64_exists = True

    with open("ex_2_32.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        if not ex_2_32_exists:
            writer.writerow(header)
        writer.writerows(res_32)

    with open("ex_2_64.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        if not ex_2_64_exists:
            writer.writerow(header)
        writer.writerows(res_64)
