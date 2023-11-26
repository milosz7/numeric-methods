import numpy as np
import matplotlib.pyplot as plt

def evaluate_node(x):
    return 1.0 / (1 + 5 * x**2)


def thomas_solve(A, b):

    x = np.zeros_like(b)
    n_dim = A.shape[0]

    for i in range(1, n_dim):
        w = A[i, i - 1] / A[i - 1, i - 1]
        A[i, i] = A[i, i] - w * A[i - 1, i]
        b[i] = b[i] - w * b[i - 1]

    x[n_dim - 1] = b[n_dim - 1] / A[n_dim - 1, n_dim - 1]

    for i in reversed(range(n_dim - 1)):
        x[i] = (b[i] - A[i + 1, i] * x[i + 1]) / A[i, i]
        
    return x


def evaluate_interpolation(x, n, nodes, node_values, xis):
    h = nodes[n+1] - nodes[n]
    a = (nodes[n+1] - x) / h
    b = (x - nodes[n]) / h
    c = (a**3 - a) * h**2 / 6
    d = (b**3 - b) * h**2 / 6
    return a * node_values[n] + b * node_values[n+1] + c * xis[n] + d * xis[n+1]


nodes = np.array([-7/8, -5/8, -3/8, -1/8, 1/8, 3/8, 5/8, 7/8], dtype="float64")
node_values = np.array([evaluate_node(x) for x in nodes], dtype="float64")
print(nodes, node_values)

n = nodes.shape[0]
xis = np.zeros(n)
h = abs(nodes[0] - nodes[1])

A = 4 * np.eye(n-2) + np.eye(n-2, k=1) + np.eye(n-2, k=(-1))
b = np.array([node_values[i] - 2 * node_values[i+1] + node_values[i+2] for i in range(n-2)])
b *= (6 / h**2)

xis[1:n-1] = thomas_solve(A, b)
print(xis)

xs = []
ys = []
x_step = 0.01

for i in range(n-1):
    x_range = np.arange(nodes[i], nodes[i+1] + x_step, step=x_step)
    for x in x_range:
        xs.append(x)
        ys.append(evaluate_interpolation(x, i, nodes, node_values, xis))

x_start = min(nodes)
x_end = max(nodes)
x_ticks_step = abs(nodes[0]) - abs(nodes[1])

plt.plot(xs, ys)
plt.scatter(nodes, node_values, c="r")
plt.xticks(np.arange(x_start, x_end + x_ticks_step, x_ticks_step))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid("both")
plt.title("Natural cubic spline interpolation results")
plt.savefig("spline_interpolation.png", dpi=400)
plt.show()