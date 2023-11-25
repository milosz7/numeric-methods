import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

def evaluate_node(x):
    return 1.0 / (1 + 5 * x**2)


def lagrange_interpolate(x, nodes, node_values, coef_precision=1e-8):
    x = sym.Symbol('x')
    fx = 0.0
    n = nodes.shape[0]
    for j in range(n):
        current_f = node_values[j]
        for i in range(n):
            if i != j:
                current_f *= (x - nodes[i])
                current_f /= (nodes[j] - nodes[i])
        fx += current_f
    fx = sym.Poly(sym.simplify(fx))
    relevant_terms = [(term, coef) for term, coef in fx.terms() if abs(coef) >= coef_precision]
    return sym.Poly(dict(relevant_terms), x)
            

nodes = np.array([-7/8, -5/8, -3/8, -1/8, 1/8, 3/8, 5/8, 7/8], dtype="float64")
node_values = np.array([evaluate_node(x) for x in nodes], dtype="float64")
print(nodes, node_values)

x_start = min(nodes)
x_end = max(nodes)
step = 0.01


f_x = lagrange_interpolate(1.0, nodes, node_values)
print("Interpolated polynomial", f_x)


xs = np.arange(x_start, x_end + step, step)
ys = np.array([f_x.eval(x) for x in xs], dtype="float64")
x_ticks_step = abs(nodes[0]) - abs(nodes[1])
plt.scatter(nodes, node_values, c="r")
plt.xticks(np.arange(x_start, x_end + x_ticks_step, x_ticks_step))
plt.plot(xs, ys)
plt.grid("both")
plt.show()
