import numpy as np
import random

def f0(x0):
    x = x0[0][0]
    y = x0[1][0]
    return 2 * x**2 + y**2 - 2


def f1(x0):
    x = x0[0][0]
    y = x0[1][0]
    return (x - 1/2)**2 + (y - 1)**2 - 1/4


def f0_dx(x0):
    x = x0[0][0]
    return 4*x


def f0_dy(x0):
    y = x0[1][0]
    return 2*y


def f1_dx(x0):
    x = x0[0][0]
    return 2*(x - 1/2)


def f1_dy(x0):
    y = x0[1][0]
    return 2 * (y - 1)


def newtons_method(x0, f0, 
                   f1, f0_dx, f0_dy, f1_dx, f1_dy, 
                   eps=1e-6, tolerance=1e-8, iterations=10000):
    x = x0
    w = 1.0
    w_shrunk = False
    for _ in range(iterations):
        if w < eps:
            return None
        if not w_shrunk:
            jac = np.array([[f0_dx(x), f0_dy(x)],
                        [f1_dx(x), f1_dy(x)]], dtype=complex)
            g_x = np.array([[f0(x)], [f1(x)]], dtype=complex)
            delta_x = np.linalg.solve(-1 * jac, g_x)
        x_n = x + w * delta_x
        grad_x = 1/2 * np.linalg.norm(g_x)
        g_n = np.array([[f0(x_n)], [f1(x_n)]], dtype=complex)
        grad_x_n = 1/2 * np.linalg.norm(g_n)
        if abs(grad_x - grad_x_n) < tolerance:
            return x_n
        if grad_x_n < grad_x:
            w = 1
            w_shrunk = False
            x = x_n
        else:
            w_shrunk = True
            w /= 2
    return None


def generate_vector():
    x = random.random() * random.randint(1, 3)
    y = random.random() * random.randint(1, 3)
    make_x_imag = random.random()
    make_y_imag = random.random()
    negate_x = random.random()
    negate_y = random.random()
    x = x * 1j if make_x_imag > 0.5 else x
    y = y * 1j if make_y_imag > 0.5 else y
    x = x * -1 if negate_x > 0.5 else x
    y = y * -1 if negate_y > 0.5 else y
    return x, y

MAX_SOLUTIONS = 4
MAX_GUESSES = 1000
solutions = []

for _ in range(MAX_GUESSES):
    isclose = False
    if len(solutions) == MAX_SOLUTIONS:
        break
    x, y = generate_vector()
    res = newtons_method(np.array([[x], [y]], dtype=complex), f0, f1, f0_dx, f0_dy, f1_dx, f1_dy)
    for sol in solutions:
        if np.allclose(res, sol):
            isclose = True
            break
    if not isclose:
        solutions.append(res)
    
print(solutions)
