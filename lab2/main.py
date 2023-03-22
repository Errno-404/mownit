import matplotlib.pyplot as plt
import numpy as np

A = -1 * np.pi
B = 2 * np.pi
POINTS = 500


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
def polynom(x, y, alg):
    points = POINTS
    a = x[0]
    b = x[len(x) - 1]

    x_axis = np.linspace(a, b, points)
    y_axis = np.zeros(points)

    for i in range(points):
        y_axis[i] = alg(x, y, x_axis[i])
    return x_axis, y_axis


# funkcja rysująca pojedynczy wykres. Przyjmuje potrzebne elementy wykresu oraz *nazwę* algorytmu, aby łatwiej
# przedstawić ją w legendzie
def draw(nodes, x, y, alg='lagrange', node='regular'):

    label="Wielomian interpolujący"
    plt.plot(x, y, color='blue', linestyle='-', label=label)
    plt.plot(x, f(x), color='red', linestyle='-', label='f(x) - funkcja interpolowana')
    plt.scatter(nodes[0], nodes[1], s=25, color='black', label='węzeł')

    plt.xlim(A, B)

    plt.xlabel('x')
    plt.ylabel('f(x)')

    alg = alg.capitalize()
    title = "Interpolacja metodą " + alg + "'a" + "\n dla węzłów "
    if node == "regular":
        title = title + "równoodległych"
    elif node == "chebyshev":
        title = title + "zgodnych z zerami wielomianu Czebyszewa"

    plt.title(title)
    plt.legend(loc='best')

    plt.show()

def estimate_error(x, y):
    return np.max(np.abs(f(x) - y))

def estimate_error2(x, y):
    return np.sqrt(np.sum((f(x) - y) ** 2)) / len(x)

# pomocnicza funkcja pozwalająca uprościć rysowanie wykresu dla poszczególnych przypadków.
def calculate_answer(node_t, alg_t, n):
    nodes = (0, 0)
    if node_t == "chebyshev":
        nodes = create_chebyshev_nodes(A, B, n)
    else:
        nodes = create_regular_nodes(A, B, n)

    if alg_t == "lagrange":
        (x, y) = polynom(nodes[0], nodes[1], lagrange)

    else:
        (x, y) = polynom(nodes[0], nodes[1], newton)


    print("Error 1: ", estimate_error(x, y))
    print("Error 2: ", estimate_error2(x, y))

    draw(nodes, x, y, alg_t, node_t)



def main():
    n = int(input('Number of nodes: '))

    # Lagrange
    calculate_answer("regular", "lagrange", n)
    calculate_answer("chebyshev", "lagrange", n)

    # Newton
    calculate_answer("regular", "newton", n)
    calculate_answer("chebyshev", "newton", n)


if __name__ == "__main__":
    main()
