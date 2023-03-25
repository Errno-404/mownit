def hermite_single_node(x, f, df):
    """
    Interpolacja Hermite'a z jednym węzłem.

    Args:
        x: Węzeł interpolacji.
        f: Wartości funkcji w węzłach.
        df: Wartości pochodnych funkcji w węzłach.

    Returns:
        Wartość interpolowanej funkcji w punkcie x.
    """
    n = len(f)
    h = [0] * n
    for i in range(n):
        h[i] = f[i]
        for j in range(i):
            h[i] = h[i] * (x - j) / (i - j) + df[j] / (i - j)
    return h[-1]



def hermite_multi_node(x, f, df):
    """
    Interpolacja Hermite'a z wieloma węzłami.

    Args:
        x: Węzeł interpolacji.
        f: Wartości funkcji w węzłach.
        df: Wartości pochodnych funkcji w węzłach.

    Returns:
        Wartość interpolowanej funkcji w punkcie x.
    """
    n = len(f)
    h = [0] * n
    for i in range(n):
        li = 1
        for j in range(n):
            if i != j:
                li *= (x - j) / (i - j)
        h[i] = li * f[i]
        for j in range(n):
            if i != j:
                h[i] /= x - j
                h[i] *= x - j - df[j]
    return sum(h)

import numpy as np




def f(x):
    return x**2

def df(x):
    return 2*x




x = np.array([1, 2])
y = f(x)
dy = df(x)


x_interp = 1.5
y_interp = hermite_multi_node(x_interp, y, dy)

print("Wartość interpolowanej funkcji w punkcie x={:.1f}: {:.3f}".format(x_interp, y_interp))
