from sympy import *
from math import pi
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate


def f_x(x):
    integral = lambda theta: sqrt(1 - x**2 * sin(theta) ** 2)
    res, _ = integrate.quad(integral, 0, pi / 2)
    return res


def expand_series(n):
    x = Symbol('x')
    result = 0
    for i in range(n):
        expr = (factorial(2*i) / (2**(2*i) * factorial(i)**2))**2 * x**(2*i) / (1-2*i)
        result += expr
    return (pi / 2) * result


def pade(coeffs, m, k):
    N = m + k
    coeffs = np.array(coeffs[:N+1][::-1])
    A = np.eye(N+1)
    for i in range(k+1, N+1):
        A[i-k:,i] = -1 * coeffs[i-k:][::-1]
    res = np.linalg.solve(A, coeffs[::-1])
    p = res[:k+1][::-1]
    p = np.poly1d(p)
    q = np.r_[1.0,res[k+1:]][::-1]
    q = np.poly1d(q)
    return p, q


n = 4
poly = Poly(expand_series(n))
poly_coeffs = [float(x) for x in poly.all_coeffs()][::-1]

p2, q2 = pade(poly_coeffs, 2, 2)
p4, q0 = pade(poly_coeffs, 4, 0)
p0, q4 = pade(poly_coeffs, 0, 4)

x = np.arange(-0.5, 0.5, 0.01)
y = [f_x(a) for a in x]
p22 = [p2(a) / q2(a) for a in x]
p40 = [p4(a) / q0(a) for a in x]
p04 = [p0(a) / q4(a) for a in x]

plt.title("E(x) plot")
plt.plot(x, y, lw=0.5, label="f(x)")
plt.savefig("function.png", dpi=400)
plt.figure()
plt.title("$R_{22}$ Padé approximation")
plt.plot(x, p22, lw=0.5, label="$R_{22}$")
plt.savefig("r22.png", dpi=400)
plt.figure()
plt.title("$R_{40}$ Padé approximation")
plt.plot(x, p40, lw=0.5, label="$R_{40}$")
plt.savefig("r40.png", dpi=400)
plt.figure()
plt.title("$R_{04}$ Padé approximation")
plt.plot(x, p22, lw=0.5, label="$R_{22}$")
plt.plot(x, p04, lw=0.5, label="$R_{04}$")
plt.savefig("r04.png", dpi=400)
plt.figure()
plt.title("Function and its Padé approximations")
plt.plot(x, p22, lw=0.5, label="$R_{22}$")
plt.plot(x, p40, lw=0.5, label="$R_{40}$")
plt.plot(x, p04, lw=0.5, label="$R_{04}$")
plt.plot(x, y, lw=0.5, label="f(x)")
plt.legend()
plt.savefig("pade_approximations.png", dpi=400)
plt.show()
