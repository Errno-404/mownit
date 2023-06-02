import numpy as np
from sys import getsizeof
from time import time

def gauss(A, B):
    n = len(A)
    time_start = time()
    for k in range(n - 1):
        # Wybieramy pivot (element o największej wartości bezwzględnej w kolumnie k)
        max_value = abs(A[k][k])
        max_index = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > max_value:
                max_value = abs(A[i][k])
                max_index = i

        # Zamiana wierszy k i max_index w macierzy A
        A[[k, max_index]] = A[[max_index, k]]

        # Zamiana elementów k i max_index w wektorze B
        B[k], B[max_index] = B[max_index], B[k]

        # Eliminacja Gaussa wierszy poniżej k-tego wiersza
        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]  # Współczynnik m
            for j in range(k, n):
                A[i][j] -= m * A[k][j]  # Modyfikacja wiersza i-tego macierzy A
            B[i] -= m * B[k]  # Modyfikacja i-tego elementu wektora B

    # Obliczenie rozwiązania x metodą wstecznej substytucji
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = np.dot(A[i][i + 1:],
                   x[i + 1:])  # Suma iloczynów elementów powyżej diagonalnej i odpowiadających im zmiennych x
        x[i] = (B[i] - s) / A[i][i]  # Obliczenie i-tej zmiennej x
    time_end = time()

    total_size = getsizeof(A) + getsizeof(A[0]) * n + getsizeof(A[0][0]) * n * n + getsizeof(B) + getsizeof(
        B[0]) * n + getsizeof(n) + getsizeof(max_index) + getsizeof(max_value) + getsizeof(m) + getsizeof(s)
    return x, total_size, time_end - time_start
