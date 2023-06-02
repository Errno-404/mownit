import numpy as np
from sys import getsizeof
from time import time

def thomas(a, b, c, d):
    time_start = time()

    n = len(d)
    w = np.zeros(n - 1)
    g = np.zeros(n)
    p = np.zeros(n)

    w[0] = c[0] / b[0]
    g[0] = d[0] / b[0]

    for i in range(1, n - 1):
        w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
    for i in range(1, n):
        g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
    p[n - 1] = g[n - 1]
    for i in range(n - 1, 0, -1):
        p[i - 1] = g[i - 1] - w[i - 1] * p[i]

    time_end = time()

    total_size = getsizeof(n) + getsizeof(w) + (n - 1) * getsizeof(w[0]) + getsizeof(g) + getsizeof(
        g[0]) * n + getsizeof(p) + getsizeof(p[0]) * n + getsizeof(a) + getsizeof(a[0]) * (n - 1) + getsizeof(
        b) + getsizeof(b[0]) * n + getsizeof(c) + getsizeof(c[0]) * (n-1) + getsizeof(d) + getsizeof(d[0]) * n

    return p, total_size, time_end - time_start
