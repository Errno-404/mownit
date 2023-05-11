from math import *


# residual
# noinspection DuplicatedCode
def secants_res(f, eps, x_n, x_n_1):
    i = 0
    while fabs(f(x_n)) > eps:
        tmp = x_n_1

        zero_check = f(x_n_1) - f(x_n)
        if zero_check <= 1e-9:
            return None, i
        x_n_1 = x_n_1 - f(x_n_1) * (x_n_1 - x_n) / (f(x_n_1) - f(x_n))
        x_n = tmp
        i += 1

    return x_n_1, i


# increasing
# noinspection DuplicatedCode
def secants_inc(f, eps, x_n, x_n_1):
    i = 0
    while fabs(x_n_1 - x_n) > eps:
        tmp = x_n_1
        # ew try-catch
        zero_check = f(x_n_1) - f(x_n)
        if zero_check <= 1e-9:
            return None, i
        x_n_1 = x_n_1 - f(x_n_1) * (x_n_1 - x_n) / (f(x_n_1) - f(x_n))
        x_n = tmp
        i += 1

    return x_n_1, i
