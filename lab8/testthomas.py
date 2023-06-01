a = [0, 1, 2, 3]
b = [2, 4, 5, 6]
c = [3, 2, 4, 0]
d = [5, 7, 8, 9]




def thomas(a, b, c, d):
    n = len(d)

    for i in range(1, n):
        w = a[i] / b[i - 1]
        b[i] = b[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]

    x = [0 for _ in range(n)]

    x[n - 1] = d[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


if __name__ == "__main__":
    print(thomas(a, b, c, d))


