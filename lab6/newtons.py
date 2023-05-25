from math import *


# residual
def newton_res(f, df, eps, x_n):
    i = 0
    while fabs(f(x_n)) > eps:
        # x_n += (direction * fun(x_n) / d_fun(x_n))
        x_n += (- f(x_n) / df(x_n))

        i += 1
    return x_n, i


# przyrostowe
def newton_inc(f, df, eps, x_n):
    i = 0
    x_n_1 = x_n + (- f(x_n) / df(x_n))
    while fabs(x_n - x_n_1) > eps:
        x_n = x_n_1
        x_n_1 += -f(x_n) / df(x_n)
        i += 1
    return x_n, i  # if x_n_1 should be returned then i + 1 should be returned as well


def solve_newtons_from_a(f, df, a, b, tol):
    roots = []
    x_curr = a
    while x_curr < b:
        roots.append([tol, x_curr, b, *newton_res(f, df, tol, x_curr), *newton_inc(f, df, tol, x_curr)])

        x_curr += 0.1
    return roots


def solve_newtons_from_b(f, df, a, b, tol):
    roots = []
    x_curr = b
    while x_curr > a:
        roots.append([tol, a, x_curr, *newton_res(f, df, tol, x_curr), *newton_inc(f, df, tol, x_curr)])

        x_curr -= 0.1
    return roots
