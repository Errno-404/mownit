from math import *


def fun(x):
    return x ** 2 - 4


def d_fun(x):
    return 2 * x


a = -1.2
b = 1.2


# residual
def newton_res(eps, x_n):
    i = 0
    while fabs(fun(x_n)) > eps:
        # x_n += (direction * fun(x_n) / d_fun(x_n))
        x_n += (- fun(x_n) / d_fun(x_n))

        i += 1
    return x_n, i


# przyrostowe
def newton_inc(eps, x_n):
    i = 0
    x_n_1 = x_n + (- fun(x_n) / d_fun(x_n))
    while fabs(x_n - x_n_1) > eps:
        x_n = x_n_1
        x_n_1 += -fun(x_n) / d_fun(x_n)
        i += 1
    return x_n, i  # if x_n_1 should be returned then i + 1 should be returned as well


if __name__ == "__main__":
    eps = float(input())
    a_ = a
    b_ = b
    while a_ < b:
        print("R: ", newton_res(eps, a))
        print("I: ", newton_inc(eps, a))
        a_ += 0.1

    while a < b_:
        print("R ", newton_res(eps, b))
        print("I ", newton_inc(eps, b))
        b_ -= 0.1
