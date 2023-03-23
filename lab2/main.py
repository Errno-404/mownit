import matplotlib.pyplot as plt
import numpy as np
import csv

# stałe wskazują przedział [A, B]
A = -1 * np.pi
B = 2 * np.pi

# stała wskazuje dla ilu punktów z przedziału [A, B] stworzony został wykres i przeprowadzona interpolacja
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
def draw(nodes, x, y, alg='lagrange', node='regular', n=0):
    label = "Wielomian interpolujący"
    plt.plot(x, y, color='blue', linestyle='-', label=label)
    plt.plot(x, f(x), color='red', linestyle='-', label='f(x) - funkcja interpolowana')
    plt.scatter(nodes[0], nodes[1], s=25, color='black', label='węzeł')

    plt.xlim(A, B)
    xticks = np.arange(-3 * np.pi / 2, 5 * np.pi / 2 + 1e-9, np.pi / 2)  # dodajemy 1e-9, aby dodać 5π/2
    xticklabels = ['-3π/2', '-π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π', '5π/2']
    plt.xticks(xticks, xticklabels)
    plt.xlabel('x')
    plt.ylabel('f(x)')

    alg = alg.capitalize()
    title = "Interpolacja metodą " + alg + "'a" + "\n dla węzłów "
    if node == "regular":
        title = title + "równoodległych"
    elif node == "chebyshev":
        title = title + "zgodnych z zerami wielomianu Czebyszewa"
    title += " dla n = " + str(n)
    plt.title(title)
    plt.legend(loc='best')

    path = "img/" + alg + '/' + node + '/' + "img_" + str(n)
    plt.savefig(path)
    plt.show()


def estimate_error(x, y):
    return np.max(np.abs(f(x) - y))


def estimate_error2(x, y):
    return np.sqrt(np.sum((f(x) - y) ** 2)) / len(x)


# pomocnicza funkcja pozwalająca uprościć rysowanie wykresu dla poszczególnych przypadków.
def calculate_answer(node_t, alg_t, n):
    if node_t == "chebyshev":
        nodes = create_chebyshev_nodes(A, B, n)
    else:
        nodes = create_regular_nodes(A, B, n)

    if alg_t == "lagrange":
        (x, y) = polynom(nodes[0], nodes[1], lagrange)

    else:
        (x, y) = polynom(nodes[0], nodes[1], newton)

    err1 = estimate_error(x, y)
    err2 = estimate_error2(x, y)
    # print("Error 1: ", err1)
    # print("Error 2: ", err2)

    draw(nodes, x, y, alg_t, node_t, n)
    return err1, err2


def main():
    best1 = (0, np.inf)
    best2 = (0, np.inf)
    best3 = (0, np.inf)
    best4 = (0, np.inf)

    res = []

    while True:
        p = input('>>')
        if p.isdigit():




            n = int(p)
            err11, val1 = calculate_answer("regular", "lagrange", n)
            err12, val2 = calculate_answer("chebyshev", "lagrange", n)
            err13, val3 = calculate_answer("regular", "newton", n)
            err14, val4 = calculate_answer("chebyshev", "newton", n)

            if val1 < best1[1]:
                best1 = n, val1
            if val2 < best2[1]:
                best2 = n, val2
            if val3 < best3[1]:
                best3 = n, val3
            if val4 < best4[1]:
                best4 = n, val4

            print(err11, val1)
            print(err12, val2)
            print(err13, val3)
            print(err14, val4)

            line = [n, err11, val1, err12, val2, err13, val3, err14, val4]
            res.append(line)
        elif p == "exit":
            break
        elif p == "results":
            print(best1, best2, best3, best4)
        else:
            print("You can either type an integer or 'exit' or 'results' commands!")


    print(best1, best2, best3, best4)
    with open('data/results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(res)

if __name__ == "__main__":
    main()
