import numpy as np
import random
import scipy.linalg as la

def laguerre_iteration(poly, z_start, tolerance=1e-10):
    n = poly.o
    der = poly.deriv()
    der2 = der.deriv()
    z_old = z_start
    while True:
        poly_z = poly(z_old)
        der_z = der(z_old)
        der2_z = der2(z_old)
        numerator = n * poly_z
        denom_rest = ((n - 1) * ((n - 1) * der_z**2 - n * poly_z * der2_z))**(1/2)
        denominator = der_z
        denominator += denom_rest if denominator > 0 else -1 * denom_rest
        z_new = z_old - numerator / denominator
        if abs(z_new - z_old) <= tolerance:
            return z_new
        z_old = z_new


def deflate_polynomial(poly, solution):
    a = poly.coef[:-1]
    A = np.eye(len(a)) - np.eye(len(a), k=(-1)) * solution
    b = la.solve_triangular(A, a, lower=True)
    return np.poly1d(b)


def find_roots(poly):
    root_poly = np.poly1d(poly.c)
    solutions = []
    while poly.o > 2:
        z_old = random.random()
        solution = laguerre_iteration(poly, z_old)
        smooth_solution = laguerre_iteration(root_poly, solution)
        solutions.append(smooth_solution)
        poly = deflate_polynomial(poly, smooth_solution)
    a, b, c = poly.coef
    delta = (b**2 - (4 * a * c))**(1/2)
    x0 = (-b - delta) / ( 2 * a)
    x1 = (-b + delta) / ( 2 * a)
    solutions.append(laguerre_iteration(root_poly, x0))
    solutions.append(laguerre_iteration(root_poly, x1))
    for sol in solutions:
        print(sol)
    print("---")
    
poly1 = np.poly1d(np.array([243, -486, 783, -990, 558, -28, -72, 16], dtype=complex))
find_roots(poly1)

poly2 = np.poly1d(np.array([1, 1, 3, 2, -1, -3, -11, -8, -12, -4, -4], dtype=complex))
find_roots(poly2)

poly3 = np.poly1d(np.array([1, 1j, -1, -1j, 1], dtype=complex))
find_roots(poly3)
