import numpy as np
import math

def find_limit(tolerance=1e-7):
    x = 0
    while (math.exp(-x) > tolerance):
        x += 1
    return x


def measure_error(approx_old, approx_new, tolerance=1e-7, epsilon=1e-5):
    return abs(approx_new - approx_old) / (abs(approx_old) + epsilon) < tolerance


def f_x(x):
    return math.sin(math.pi * (1.0 + math.sqrt(x)) / (1.0 + x**2)) * math.exp(-x)


def calculate_integral(x_start, x_end):
    approx_old = 1e10
    n_elements = 2
    constant_factor = (f_x(x_start) + f_x(x_end)) / 2.0

    while True:
        splits = np.linspace(x_start, x_end, n_elements)
        approx_new = constant_factor
        jump = abs(splits[0] - splits[1])
        for i in range(1, len(splits) - 1):
            approx_new += f_x(splits[i])
        approx_new *= jump
        if measure_error(approx_old, approx_new):
            print("Integration result is:", approx_new)
            break
        n_elements += (n_elements - 1)
        approx_old = approx_new


x_start = 0.0
x_end = find_limit()

calculate_integral(x_start, x_end)
