import matplotlib.pyplot as plt
import numpy as np

f_x = lambda x: 1/4 * x**4 - 1/2 * x**2 - 1/16 * x

def swap_gss(f, a, b, c, d):
    f_b = f(b)
    f_d = f(d)
    if f_d < f_b:
        if d < b:
            c = b
            b = d
        else:
            a = b
            b = d
    else:
        if d < b:
            a = d
        else:
            c = d

    return a, b, c, d


def gss_iteration(f, a, b, c):
    w = (3 - 5**(1/2)) / 2
    b_a = abs(b - a)
    b_c = abs(b - c)

    if b_a > b_c:
        d = a + w * b_a
    else:
        d = b + w * b_c
    b_old = b

    a, b, c, d = swap_gss(f, a, b, c, d)
    return a, b, c, d, b_old


def golden_search(f, a, c, tolerance=1e-6, max_iterations=10000):
    intervals = []
    iterations = 0
    b = (a + c) / 2

    while True:
        a, b, c, d, b_old = gss_iteration(f, a, b, c)
        a_c = abs(c - a)
        iterations += 1
        intervals.append(a_c)
        if a_c < tolerance * (abs(b_old) + abs(d)):
            break
        if iterations == max_iterations:
            print("Convergence failed!")
            return None

    print("gss","[", a, c, "]", "converged in", iterations)
    return iterations, intervals


def swap_brent(f, a, b, c, d, f_a, f_b, f_c):
    f_d = f(d)
    if f_d < f_b:
        if d < b:
            c, f_c = b, f_b
            b, f_b = d, f_d
        else:
            a, f_a = b, f_b
            b, f_b = d, f_d
    else:
        if d < b:
            a, f_a = d, f_d
        else:
            c, f_c = d, f_d

    return a, b, c, d, f_a, f_b, f_c


def brent_iteration(f, a, b, c, f_a, f_b, f_c):
    d = 1/2 * ((a**2 * (f_c - f_b)) + b**2 * (f_a - f_c) + c**2 * (f_b - f_a)) / \
        (a * (f_c - f_b) + b * (f_a - f_c) + c * (f_b - f_a))
    b_old = b
    if not (a < d and d < c):
        d = (a + c) / 2

    a, b, c, d, f_a, f_b, f_c = swap_brent(f, a, b, c, d, f_a, f_b, f_c)
    return  a, b, c, d, f_a, f_b, f_c, b_old


def brent_method(f, a, b, tolerance=1e-6, max_iterations=10000):
    intervals = []
    iterations = 0
    c = (a + b) / 2
    f_a, f_b, f_c = f_x(a), f_x(b), f_x(c)
    
    while True:
        a, b, c, d, f_a, f_b, f_c, b_old = brent_iteration(f, a, b, c, f_a, f_b, f_c)
        a_c = abs(c - a)
        intervals.append(a_c)
        iterations += 1
        if a_c < tolerance * (abs(b_old) + abs(d)):
            break
        if iterations == max_iterations:
            print("Convergence failed!")
            return None
        
    print("brent","[", a, c, "]", "converged in", iterations)
    return iterations, intervals


a_start = 0
b_start = 5
plt.title(f"Comparision of convergence between methods (starting interval: [{a_start}, {b_start}])")
plt.xlabel("Iterations")
plt.ylabel("abs(a-c)")
iterations, intervals = golden_search(f_x, a_start, b_start)
plt.plot(np.arange(0, iterations), intervals, label="Golden section search")
iterations, intervals = brent_method(f_x, a_start, b_start)
plt.plot(np.arange(0, iterations), intervals, label="Brent method")
plt.legend()
plt.savefig("convergence_comparision.png", dpi=400)
plt.show()
