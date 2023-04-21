# Functions are written in that specific manner, so they can be easily converted to a class methods when I learn how to.
import numpy as np
import os
import csv
from matplotlib import pyplot as plt

A = -np.pi
B = 2 * np.pi
P = 1000
IMG_PATH = "./img"
ERRORS_PATH = "./data"


def fun(x, m=3, k=3):
    return np.exp(-k * np.sin(m * x)) + k * np.sin(m * x) - 1


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
    return estimate_error2(f, xaxis, yaxis)


def estimate_error2(f, x, y):
    return np.sqrt(np.sum((f(x) - y) ** 2)) / len(x)


def draw(x, xaxis, yaxis, f, m, n):
    title = f"Aproksymacja dla n = {n} i m = {m}"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xaxis, f(xaxis), label="F(x)")
    plt.plot(xaxis, yaxis, label="f(x)")
    plt.scatter(x, f(x), label="węzeł")
    plt.legend(loc='best')
    plt.savefig(IMG_PATH + f"/img_{m}_{n}")
    plt.show()


def main():
    n = m = 0
    ready = False

    # creating directories if they don't exist
    if not os.path.exists(ERRORS_PATH):
        os.makedirs(ERRORS_PATH)

    # adding a header if file does not exist
    if not os.path.isfile(ERRORS_PATH + "/errors.csv"):
        with open(ERRORS_PATH + "/errors.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["m", "n", "error"])

    # creating directory for images
    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)

    data = []

    # repl
    while True:
        print("(m, n) = ({}, {})".format(m, n))

        user = input("Choose option: n | m | run | q\n>")
        if user == "n":

            while True:

                print("(m, n) = ({}, {})".format(m, n))
                user = input("> n = ")
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
                print("(m, n) = ({}, {})".format(m, n))
                user = input("m = ")
                if user.isnumeric() and int(user) > 0:
                    tmp = int(user)
                    if 0 < tmp < n:
                        ready = True
                        m = tmp
                        break
                    elif 0 < tmp == n:
                        confirmation = input("You are going to interpolate because n = m. Are you sure?(y/n)\n>")
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

        elif user == "q":
            with open(ERRORS_PATH + "/errors.csv", 'a', newline='') as file:
                writer = csv.writer(file)

                # write data row by row
                if len(data) > 0:
                    for row in data:
                        # print(row)
                        writer.writerow(row)
            return
        elif user == "run":
            if ready:
                print("running...")

                x = np.linspace(A, B, n + 1)
                w = [1 for _ in range(n + 1)]
                error = (calculate(m, fun, n, w, x))
                row = [m, n, error]
                data.append(row)

            else:
                print("Cannot run program with parameters (m, n) = ({}, {})!".format(m, n))
        else:
            print("Unknown command!")


if __name__ == "__main__":
    main()
