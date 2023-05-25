import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return x ** 15 + x ** 12


xaxis = np.linspace(-0.05, 0.05, 1000)
plt.plot(xaxis, f(xaxis), label="f(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Przybliżenie na okolice x = 0")
plt.legend(loc="best")
plt.show()
