import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, linewidth=150)

nodes = []
values = []
with open("ex15_data.txt") as f:
    data = f.read().splitlines()
    for line in data:
        x, y = line.split()
        nodes.append(float(x))
        values.append(float(y))

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


N = len(nodes)
xis = np.zeros(N, dtype="float64")
A = np.zeros((N-2, N-2), dtype="float64")

bs = []
for i in range(0, N-2):
    for j in range(i, i+2):
        diff_0 = nodes[j] - nodes[j-1]
        diff_1 = nodes[j+1] - nodes[j-1]
        diff_2 = nodes[j+1] - nodes[j]
        f_diff_0 = values[j+1] - values[j]
        f_diff_1 = values[j] - values[j-1]
    if i > 0:
        A[i, i-1] = diff_0 / 6
    A[i, i] = diff_1 / 3
    if i < N-3:
        A[i, i+1] = diff_2 / 6
    bs.append(f_diff_0 / diff_2 - f_diff_1 / diff_0)

bs = np.array(bs)
xis[1:N-1] = thomas_solve(A, bs)

print(A)
print(xis)
xs = []
ys = []
for i in range(N-1):
    x_range = np.linspace(nodes[i], nodes[i+1], 5)
    for x in x_range:
        xs.append(x)
        ys.append(evaluate_interpolation(x, i, nodes, values, xis))

plt.title("Cubic spline for range [-1.5, 1.5]")
plt.plot(xs, ys, label="cubic spline")
# plt.scatter(nodes, values, c="r", s=0.5, label="points used for interpolation")
plt.legend()
plt.show()