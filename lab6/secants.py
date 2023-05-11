from math import *


def fun(x):
    return x ** 5 + x ** 4 - 2 * x - 5


a = -1.2
b = 1.2


def secants_res(eps, x_n, x_n_1):
    i = 0
    while fabs(fun(x_n)) > eps:
        tmp = x_n_1

        # ew try-catch
        t = fun(x_n_1) - fun(x_n)
        if t == 0:
            return x_n_1, i, "exception"

        x_n_1 = x_n_1 - fun(x_n_1) * (x_n_1 - x_n) / (fun(x_n_1) - fun(x_n))
        x_n = tmp
        i += 1

    return x_n_1, i


def secants_inc(eps, x_n, x_n_1):
    i = 0
    while fabs(x_n_1 - x_n) > eps:
        tmp = x_n_1

        # ew try-catch
        t = fun(x_n_1) - fun(x_n)
        if t == 0:
            return x_n_1, i, "exception"

        x_n_1 = x_n_1 - fun(x_n_1) * (x_n_1 - x_n) / (fun(x_n_1) - fun(x_n))
        x_n = tmp
        i += 1

    return x_n_1, i


# jeszcze przemyśleć zamianę a z b i dorobić wersję inc

# idk czy tu mozna b z a zamienić ? chyb nie chociaż działało i może jednak można ?
print(secants_inc(0.00000001, b, a))
