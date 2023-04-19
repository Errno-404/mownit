# Functions are written in that specific manner, so they can be easily converted to a class methods when I learn how to.
import numpy as np
from matplotlib import pyplot as plt

A = -np.pi
B = 2 * np.pi
P = 200


def fun(x):
    return np.sin(x)


def get_weight(w, i):
    return w[i]


def get_x(x, i):
    return x[i]


def get_coef(k, n, m, w, x):
    ret = 0
    for i in range(n + 1):
        ret += get_weight(w, i) * get_x(x, i) ** (m + k)

    return ret


def get_constants(f, k, n, w, x):
    ret = 0
    for i in range(n + 1):
        ret += (w[i] * f(get_x(x, i)) * (get_x(x, i)) ** k)
    return ret


def solve_matrix(f, n, m, w, x):
    matrix = np.zeros((m + 1, m + 1))
    constants = np.zeros(m + 1)
    for i in range(m + 1):
        for j in range(m + 1):
            matrix[i][j] = get_coef(i, n, j, w, x)

        constants[i] = get_constants(f, i, n, w, x)

    a = np.linalg.solve(matrix, constants)
    return a


def get_polynom(x0, m, a):
    ret = 0
    for i in range(m + 1):
        ret += a[i] * x0 ** i

    return ret


def calculate(m, f, n, w, x):
    xaxis = np.linspace(A, B, P)
    yaxis = np.zeros(P)

    a = solve_matrix(f, n, m, w, x)

    for i in range(P):
        yaxis[i] = get_polynom(xaxis[i], m, a)
    draw(x, xaxis, yaxis, f, m, n)


def repl():
    n = m = 0

    ready = False
    while True:
        print("(n, m) = ({}, {})".format(n, m))
        if ready:
            print(ready)




        ready = False
        user = input("n/m/exit")
        if user == "n":

            while True:




                print("(n, m) = ({}, {})".format(n, m))
                user = input("n = ")
                if user.isnumeric() and int(user) > 0:
                    tmp = int(user)
                    if 0 < m < tmp:
                        ready = True
                        n = tmp
                        break
                    elif 0 < m == tmp:
                        confirmation = input("You are going to interpolate because n = m. Are you sure?(y/n)")
                        if confirmation == "y":
                            ready = True
                            n = tmp
                            break
                    elif 0 < tmp < m:
                        print("n cannot be smaller than m!")
                    elif m == 0:
                        n = tmp
                        break

                elif user.isnumeric():
                    print("n must be bigger than 0!")
                elif user == "q":
                    # hidden escape
                    break
                else:
                    print("n is not a number!")

        elif user == "m":
            while True:
                print("(n, m) = ({}, {})".format(n, m))
                user = input("m = ")
                if user.isnumeric() and int(user) > 0:
                    tmp = int(user)
                    if 0 < tmp < n:
                        ready = True
                        m = tmp
                        break
                    elif 0 < tmp == n:
                        confirmation = input("You are going to interpolate because n = m. Are you sure?(y/n)")
                        if confirmation == "y":
                            ready = True
                            m = tmp
                            break
                    elif 0 < n < tmp:
                        print("n cannot be smaller than m!")
                    elif n == 0:
                        m = tmp
                        break

                elif user.isnumeric():
                    print("n must be bigger than 0!")
                elif user == "q":
                    # hidden escape
                    break
                else:
                    print("n is not a number!")
        elif ready is True:
            print(True)





def draw(x, xaxis, yaxis, f, m, n):
    plt.plot(xaxis, f(xaxis))
    plt.plot(xaxis, yaxis)
    plt.scatter(x, f(x))
    plt.show()


def main():
    n = 50
    m = 6
    x = np.linspace(A, B, n + 1)
    # x = [0, 1, 2, 3, 4]
    w = [1 for _ in range(n + 1)]
    calculate(m, fun, n, w, x)


if __name__ == "__main__":
    # main()
    repl()
