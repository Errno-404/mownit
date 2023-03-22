import matplotlib.pyplot as plt
import numpy as np


# funkcja interpolowana
def f(x):
    m = 3
    k = 3

    return np.exp(-k * np.sin(m * x)) + k * np.sin(m * x) - 1


# funkcja tworzy węzły zgodne z zerami wielomianu Czebyszewa
def create_chebyshev_nodes(a, b, n):
    x = np.zeros(n)
    y = np.zeros(n)

    for k in range(1, n + 1):
        x[k - 1] = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(((2 * k - 1) / (2 * n)) * np.pi)
        y[k - 1] = f(x[k - 1])

    return x, y


# funkcja tworzy węzły równomiernie rozłożone na przedziale (a, b)
def create_regular_nodes(a, b, n):
    x = np.linspace(a, b, n)
    y = np.zeros(n)

    for i in range(n):
        y[i] = f(x[i])

    return x, y


# funkcja realizująca interpolację wzorem Lagrange'a dla funkcji f(x) w punkcie x = x0.
# Dane są natomiast węzły (x, y)
def lagrange(x, y, x0):
    n = len(x)
    p = 0

    for k in range(n):

        l_k = 1
        for i in range(n):
            if k == i:
                continue

            l_k *= (x0 - x[i]) / (x[k] - x[i])

        p += l_k * y[k]
    return p


# Funkcja pomocnicza zwracająca tablicę wypełnioną ilorazami różnicowymi potrzebną w metodzie Newtona
def diff_quotients(x, y):
    n = len(x)
    dp = np.zeros((n, n))

    for i in range(n):
        dp[i][0] = y[i]

    for i in range(1, n):
        ind = 0
        for j in range(i, n):
            dp[j][i] = (dp[j][i - 1] - dp[j - 1][i - 1]) / (x[j] - x[ind])
            ind += 1
    return dp


# funkcja realizująca interpolację wzorem Lagrange'a dla funkcji f(x) w punkcie x = x0.
# Dane są natomiast węzły (x, y)
def newton(x, y, x0):
    n = len(x)

    b_k = diff_quotients(x, y)
    w = b_k[0][0]
    for k in range(1, n):
        p_k = 1
        for i in range(k + 1):
            if i < k:
                p_k *= (x0 - x[i])

        w += p_k * b_k[k][k]
    return w


# Funkcja wykonuje interpolację na przedziale (a, b) zawartym w zmiennej x, gdzie (x, y) to węzły, natomiast alg to
# odpowiedni algorytm interpolujący. Do funkcji można przekazać dodatkowo liczbę punktów points > n, na podstawie
# której będzie można narysować gładki wykres.
def polynom(x, y, alg, points=500):
    points = 500
    a = x[0]
    b = x[len(x) - 1]

    x_axis = np.linspace(a, b, points)
    y_axis = np.zeros(points)

    for i in range(points):
        y_axis[i] = alg(x, y, x_axis[i])
    return x_axis, y_axis


def draw(nodes, x, y):
    plt.plot(x, y)
    plt.plot(x, f(x))
    plt.show()


def draw_common(x_l, y_l, x_c, y_c, x_r, y_r, x_cc, y_cc):
    plt.scatter(x_r, y_r, color="red")
    plt.plot(x_l, y_l)
    plt.plot(x_c, y_c)

    plt.scatter(x_cc, y_cc)

    plt.show()


def main():
    # parameters to change
    a = -1 * np.pi
    b = 2 * np.pi
    n = 8

    t = create_regular_nodes(a, b, n)
    (k, l) = polynom(t[0], t[1], newton)
    draw(t, k, l)

    # # choose one of them
    # alg = lagrange
    # alg2 = newton
    #
    # x_r, y_r = create_regular_nodes(a, b, n)
    # x_c, y_c = create_chebyshev_nodes(a, b, n)
    #
    # # Lagrange:
    # ax_l, ay_l = polynom(x_r, y_r, lagrange)
    # ax_l_c, ay_l_c = polynom(x_c, y_c, lagrange)
    #
    # # Newton
    # ax_n_c, ay_n_c = polynom(x_c, y_c, newton)
    # ax_n, ay_n = polynom(x_r, y_r, newton)
    #
    # draw(ax_l, ay_l, x_r, y_r)
    # draw(ax_n, ay_n, x_r, y_r)
    #
    # draw(ax_l_c, ay_l_c, x_c, y_c)
    # draw(ax_n_c, ay_n_c, x_c, y_c)
    #
    # draw_common(ax_l, ay_l, ax_l_c, ay_l_c, x_r, y_r, x_c, y_c)
    # draw_common(ax_n, ay_n, ax_n_c, ay_n_c, x_r, y_r, x_c, y_c)


if __name__ == "__main__":
    main()
