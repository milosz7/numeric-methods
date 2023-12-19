import numpy as np
import math
import matplotlib.pyplot as plt
import sys

plt.rcParams.update({
    "text.usetex": True
})

def find_limit(tolerance=1e-8):
    x = 0
    while (math.exp(-(x**2)) > tolerance):
        x += 1
    return x


def f_x(x):
    return math.cos((1 + x) / (x**2 + 0.04)) * math.exp(-(x**2))


def trapeze_method(a, b):
    return (b - a) / 2 * (f_x(a) + f_x(b))


def pack(a, b, result):
    return ((a,b), result)


def calculate_approx(left_result, right_result, result):
    return  (4 *(left_result + right_result) - result) / 3
    

def error_below_tolerance(left_result, right_result, result, tolerance=1e-8):
    return abs((left_result + right_result - result) / 3) < tolerance


MAX_STACK_SIZE = 1000

x_end = find_limit()
x_start = x_end * -1
x_middle = (x_end + x_start) / 2
i_all = trapeze_method(x_start, x_end)

print("integration limits", x_start, x_end)

stack = []
xs = []
ys = []
result = 0

while True:
    i_left = trapeze_method(x_start, x_middle)
    i_right = trapeze_method(x_middle, x_end)

    if not error_below_tolerance(i_left, i_right, i_all):
        stack.append(pack(x_middle, x_end, i_right))
        x_end = x_middle
        x_middle = (x_start + x_end) / 2
        i_all = i_left
    else:
        xs.append(x_middle)
        result += calculate_approx(i_left, i_right, i_all)
        ys.append(result)
        if not stack:
            break
        (a, b), i = stack.pop()
        x_start = a
        x_end = b
        x_middle = (x_start + x_end) / 2
        i_all = i
    if (len(stack) >= MAX_STACK_SIZE):
        print("Too big of a precision for max stack size.")
        sys.exit(1)

xs0 = np.linspace(-2, 2, 500)
plt.plot(xs0, [f_x(x) for x in xs0], lw=0.7)
plt.title("$\\frac{cos(1+x)}{0.04 + x^2}e^{-x^2}$ plot")
plt.grid("both")
plt.savefig("ex12_function_plot.png", dpi=400)

plt.figure()
plt.title("$\int_{-5}^{5}\\frac{cos(1+x)}{0.04 + x^2}e^{-x^2}dx$ plot")   
ticks = np.arange(-5, 5.5, step=1)
plt.xticks(ticks)
plt.plot(xs, ys, lw=0.7)
plt.grid("both")
plt.savefig("ex12_antiderivative_plot.png", dpi=400)

print("Integration result:", result)
