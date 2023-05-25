from newtons import *
from secants import *
import csv
import os

a = -1.2
b = 1.2


def fun(x):
    return x ** 15 + x ** 12


def d_fun(x):
    return 15 * x ** 14 + 12 * x ** 11


if __name__ == "__main__":
    eps = float(input())

    secants_from_a_left, secants_from_a_right = (solve_secants_from_a(fun, a, b, eps))
    secants_from_b_left, secants_from_b_right = (solve_secants_from_b(fun, a, b, eps))

    print(secants_from_a_left)

    newtons_left = solve_newtons_from_a(fun, d_fun, a, b, eps)
    newtons_right = solve_newtons_from_b(fun, d_fun, a, b, eps)

    if not os.path.exists("./data"):
        os.mkdir("./data")

    if not os.path.isfile("./data/newtons_left.csv"):
        newtons_left = [["eps", "a", "b", "res_root", "iter", "inc_root", "iter"]] + newtons_left
    if not os.path.isfile("./data/newtons_right.csv"):
        newtons_right = [["eps", "a", "b", "res_root", "iter", "inc_root", "iter"]] + newtons_right

    # files for secants
    if not os.path.isfile("./data/secants_from_a_left.csv"):
        secants_from_a_left = [["eps", "a", "b", "res_root", "iter", "inc_root", "iter"]] + secants_from_a_left
    if not os.path.isfile("./data/secants_from_a_right.csv"):
        secants_from_a_right = [["eps", "a", "b", "res_root", "iter", "inc_root", "iter"]] + secants_from_a_right
    if not os.path.isfile("./data/secants_from_b_left.csv"):
        secants_from_b_left = [["eps", "a", "b", "res_root", "iter", "inc_root", "iter"]] + secants_from_b_left
    if not os.path.isfile("./data/secants_from_b_right.csv"):
        secants_from_b_right = [["eps", "a", "b", "res_root", "iter", "inc_root", "iter"]] + secants_from_b_right

    with open("data/newtons_left.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(newtons_left)
    with open("data/newtons_right.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(newtons_right)

    with open("data/secants_from_a_left.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(secants_from_a_left)
    with open("data/secants_from_a_right.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(secants_from_a_right)

    with open("data/secants_from_b_left.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(secants_from_b_left)
    with open("data/secants_from_b_right.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(secants_from_b_right)
