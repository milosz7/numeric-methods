import numpy as np
import matplotlib.pyplot as plt
import math

np.set_printoptions(suppress=True, linewidth=150)

def generate_vandermonde(xs):
    n = xs.shape[0]
    vandermonde = np.empty((n, n), dtype="float64")
    for i in range(n):
        vandermonde[i, :n] = [math.pow(xs[i], k) for k in reversed(range(n))]
    return vandermonde


def evaluate_function(coefficients, x):
    n = coefficients.shape[0]
    highest_pow = n - 1
    output = 0.0
    for i in range(n):
        output += coefficients[i] * math.pow(x, highest_pow - i)
    return output


nodes = np.array([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.00], dtype="float64")
node_values = np.array([1.13092041015625,
                        2.3203125,
                        1.92840576171875,
                        1.0,
                        0.05548095703125,
                        -0.6015625, 
                        -0.75250244140625,
                        0.0], dtype="float64")
x_start = -1.25
x_end = 1.25
x_step = 0.01

nodes_vandermonde = generate_vandermonde(nodes)

coefficients = np.linalg.solve(nodes_vandermonde, node_values)
print(nodes_vandermonde)
print(coefficients)

xs = np.arange(start=x_start, stop=x_end + x_step, step=x_step)
ys = [evaluate_function(coefficients, x) for x in xs]

plt.title("Function interpolation results")
plt.xlabel("x")
plt.xticks(np.arange(x_start, x_end+0.25, step=0.25))
plt.yticks(np.arange(-20, 5, 1))
plt.grid("both")
plt.ylabel("$f_x$")
plt.plot(xs, ys, label="Interpolated function plot")
plt.scatter(nodes, node_values, c="red", label="Points used for interpolation")
plt.legend()
plt.savefig("function_interpolation.png", dpi=400)
plt.show()
