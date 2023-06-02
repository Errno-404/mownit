from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot


def jacobi(A, b, N=25, x0=None):
    # Create an initial guess if needed
    if x0 is None:
        x0 = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = diag(A)
    R = A - diagflat(D)

    # Iterate for N times                                                                                                                                                                          
    for i in range(N):
        x0 = (b - dot(R, x0)) / D
    return x0

import numpy as np





A = array([[2.0, 1.0], [5.0, 7.0]])
b = array([11.0, 13.0])
guess = array([1.0, 1.0])

sol = jacobi(A, b, N=25, x0=guess)

print("A:")
pprint(A)

print("b:")
pprint(b)

print("x:")
pprint(sol)
