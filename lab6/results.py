from newtons import *
from secants import *
import csv
import os

a = -1.2
b = 1.2


def fun(x):
    return x ** 5 + x ** 4 - 2 * x - 5


def d_fun(x):
    return 5 * x ** 4 + 4 * x ** 3 - 2


if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.mkdir("./data")

    eps = float(input())
    a_ = a
    b_ = b

    newtons_left = [["eps", "a", "b", "res_root", "iter", "inc_root", "iter"]]
    secants_left_left = [["eps", "a", "b", "res_root", "iter", "inc_root", "iter"]]  # start from a [a, a_]
    secants_left_right = [["eps", "a", "b", "res_root", "iter", "inc_root", "iter"]]  # start from a [a_, b]
    while a_ < b:
        # newton from left
        # noinspection DuplicatedCode
        newtons_left.append([eps, a_, b, *newton_res(fun, d_fun, eps, a_), *newton_inc(fun, d_fun, eps, a_)])

        # secants from left
        secants_left_left.append([eps, a, a_, *secants_res(fun, eps, a, a_), *secants_inc(fun, eps, a, a_)])
        secants_left_right.append([eps, a_, b, *secants_res(fun, eps, a_, b), *secants_inc(fun, eps, a_, b)])
        a_ += 0.1

    newtons_right = [["eps", "a", "b", "res_root", "iter", "inc_root", "iter"]]
    secants_right_left = [["eps", "a", "b", "res_root", "iter", "inc_root", "iter"]]
    secants_right_right = [["eps", "a", "b", "res_root", "iter", "inc_root", "iter"]]
    while a < b_:
        # noinspection DuplicatedCode
        newtons_right.append([eps, a, b_, *newton_res(fun, d_fun, eps, b_), *newton_inc(fun, d_fun, eps, b_)])
        secants_right_left.append([eps, a, b_, *secants_res(fun, eps, a, b_), *secants_inc(fun, eps, a, b_)])
        secants_right_right.append([eps, b_, b, *secants_res(fun, eps, b_, b), *secants_inc(fun, eps, b_, b)])

        b_ -= 0.1

    with open("data/newtons_left.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(newtons_left)
    with open("data/newtons_right.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(newtons_right)

    with open("data/secants_left_left.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(secants_left_left)
    with open("data/secants_left_right.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(secants_left_right)

    with open("data/secants_right_left.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(secants_right_left)
    with open("data/secants_right_right.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(secants_right_right)

    print(newtons_left)
    print(newtons_right)

    print(secants_left_left)
    print(secants_left_right)

    print(secants_right_left)
    print(secants_right_right)
