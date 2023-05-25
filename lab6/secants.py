import math


def secants_inc(f, a, b, tol):
    x_prev = a
    x_curr = b
    i = 0

    while not math.fabs(x_prev - x_curr) < tol:
        x_next = x_curr - f(x_curr) * (x_curr - x_prev) / (f(x_curr) - f(x_prev))
        x_prev = x_curr
        x_curr = x_next
        i += 1

    return x_curr, i


def secants_res(f, a, b, tol):
    x_prev = a
    x_curr = b
    i = 0

    while not math.fabs(f(x_curr)) < tol:
        x_next = x_curr - f(x_curr) * (x_curr - x_prev) / (f(x_curr) - f(x_prev))
        x_prev = x_curr
        x_curr = x_next
        i += 1

    return x_curr, i


# ze zwiększającym się a

def solve_secants_from_a(f, a, b, tol):
    roots_left = []
    roots_right = []
    x_curr = a + 0.1
    while x_curr < b:
        roots_left.append([tol, a, x_curr, *secants_res(f, a, x_curr, tol), *secants_inc(f, a, x_curr, tol)])
        roots_right.append([tol, x_curr, b, *secants_res(f, x_curr, b, tol), *secants_inc(f, x_curr, b, tol)])

        x_curr += 0.1
    return roots_left, roots_right


def solve_secants_from_b(f, a, b, tol):
    roots_left = []
    roots_right = []
    x_curr = b - 0.1
    while x_curr > a:
        roots_left.append([tol, a, x_curr, *secants_res(f, a, x_curr, tol), *secants_inc(f, a, x_curr, tol)])
        roots_right.append([tol, x_curr, b, *secants_res(f, x_curr, b, tol), *secants_inc(f, x_curr, b, tol)])

        x_curr -= 0.1
    return roots_left, roots_right
