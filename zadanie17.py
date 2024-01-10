import numpy as np
import random
import matplotlib.pyplot as plt

f = lambda x, y : (1 - x)**2 + 100 * (y - x**2)**2
f_dx = lambda x, y: 2 * (200 * x**3 - 200 * x * y + x - 1)
f_dy = lambda x, y: 200 * (y - x**2)
f_dxdx = lambda x, y : 1200 * x**2 - 400 * y + 2
f_dydy = 200;
f_dydx = lambda x : -400 * x

def levenberg_marquardt(f, f_dx, f_dy, f_dxdx, f_dydy, f_dydx, x0, iteration, coeff=2e-10, tolerance=1e-6, multiplier=8, max_coeff = 1e6):
    x, y = x0
    x_old = x0
    f_old = f(x, y)
    gradient = np.array([f_dx(x, y), f_dy(x, y)], dtype="float64")
    steps = 0
    xs = [x]
    ys = [y]
    
    while np.linalg.norm(gradient) > tolerance:
        if coeff > max_coeff:
            print(f"Convergence impossible for {x0}.")
            return None

        hessian = np.array([[f_dxdx(x, y), f_dydx(x)], 
                            [f_dydx(x), f_dydy]], dtype="float64")
        hessian[0][0] *=  (1 + coeff)
        hessian[1][1] *= (1 + coeff)
        new_pair = np.linalg.solve(hessian, gradient)
        new_pair = x_old - new_pair
        new_x, new_y = new_pair

        f_new = f(new_x, new_y)
        if f_new > f_old:
            coeff *= multiplier
        else:
            xs.append(new_x)
            ys.append(new_y)
            steps += 1
            coeff /= multiplier
            x_old = new_pair
            x, y = x_old
            f_old = f(x, y)
            gradient = np.array([f_dx(x, y), f_dy(x, y)], dtype="float64")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_box_aspect(None, zoom=0.9)
    X = np.arange(-2, 2, 0.25)
    Y = np.arange(-2, 6, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)
    ax.set_zlim(0, 4000)
    ax.plot_wireframe(X, Y, Z, linewidth=0.5, alpha=0.4, color="gray")

    zs = [f(x, y) for (x, y) in zip(xs, ys)]
    ax.scatter(xs, ys, zs, s=10, c="r", label="intermediate steps")
    ax.plot(xs, ys, zs, c="r")
    ax.scatter(xs[0], ys[0], zs[0], label=f"first point ({xs[0]:.6f}, {ys[0]:.6f})", s=50)
    ax.scatter(xs[-1], ys[-1], zs[-1], label=f"last point ({xs[-1]:.6f}, {ys[-1]:.6f})", s=50)
    ax.view_init(30, 50, 0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.legend()
    fig.savefig(f"levenberg_marquardt_3d_{iteration}.png", dpi=400)

    fig2d = plt.figure()
    plt.plot(xs, ys)
    plt.scatter(xs, ys, c="r")
    plt.title(f"2D plot of path (start: ({xs[0]:.6f}, {ys[0]:.6f}))")
    fig2d.savefig(f"levenberg_marquardt_2d_{iteration}.png", dpi=400)

    print("Found minimum:", x, y, "steps", steps)


points_to_generate = 5
max_x = 2
min_x = -2
max_y = 6
min_y = -2

for i in range(points_to_generate):
    point = np.array([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
    levenberg_marquardt(f, f_dx, f_dy, f_dxdx, f_dydy, f_dydx, point, i)
