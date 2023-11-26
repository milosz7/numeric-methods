# https://www.inf.usi.ch/hormann/papers/Hormann.2016.PAF.pdf
# alternative approach for indexing (line 17)
# J_k = [i for i in range(k - d, k+1) if i >= 0 and i < n - d + 1]
# for i in J_k:

import numpy as np
import matplotlib.pyplot as plt

def evaluate_node(x):
    return 1.0 / (1 + 5 * x**2)


def calculate_weights(nodes, d):
    n = nodes.shape[0]
    weights = np.zeros(n, dtype="float64")
    for k in range(n):
        for i in range(max(0, k-d), min(k, n-d) + 1):
            multipliter = 1.0 if i % 2 == 0 else -1.0
            acc = 1.0
            for j in range(i, i + d):
                if j != k:
                    acc *= 1 / (nodes[j] - nodes[k])
            weights[k] += multipliter * acc 
    return weights


def interpolate_value(x, weights, nodes, node_values):
    if x in nodes:
        return evaluate_node(x)
    n = nodes.shape[0]
    weight_sum = 0.0
    output = 0.0
    for i in range(n):
        weight = (weights[i]) / (x - nodes[i])
        weight_sum += weight
        output += weight * node_values[i]
    return output / weight_sum


d = 3
nodes = np.array([-7/8, -5/8, -3/8, -1/8, 1/8, 3/8, 5/8, 7/8], dtype="float64")
node_values = np.array([evaluate_node(x) for x in nodes], dtype="float64")
weights = calculate_weights(nodes, d)

x_start = min(nodes)
x_end = max(nodes)
step = 0.01

xs = np.arange(x_start, x_end + step, step) 
ys = [interpolate_value(x, weights, nodes, node_values) for x in xs]
x_ticks_step = abs(nodes[0]) - abs(nodes[1])

plt.scatter(nodes, node_values, c="r", label="Nodes used for interpolation")
plt.xticks(np.arange(x_start, x_end + x_ticks_step, x_ticks_step))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(xs, ys, label="Interpolated function plot")
plt.grid("both")
plt.legend()
plt.title("Hormann-Floater interpolation results")
plt.savefig("hormann_interpolation.png", dpi=400)
plt.show()