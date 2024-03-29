import numpy as np

f = lambda x, y, z, v : (1-x)**2 + 100*(y-x**2)**2 + 100*(z-y**2)**2 + 100*(v-z**2)**2
f_dx = lambda x, y : 400*x**3 - 400*x*y + 2*x - 2
f_dy = lambda x, y, z : -200*x**2 + 400*y**3 - 400*y*z + 200*y
f_dz = lambda y, z, v : -400*v*z - 200*y**2 + 400*z**3 + 200*z
f_dv = lambda z, v : 200 * (v -z**2)


def gradient_descent(f, f_dx, f_dy, f_dz, f_dv, pt0, lr=1e-4, lr_decay = 0.9999, tolerance=1e-4):
    x, y, z, v = pt0
    pt_old = pt0
    f_old = f(x, y, z, v)
    gradient = np.array([f_dx(x, y), f_dy(x, y, z), f_dz(y, z, v), f_dv(z, v)], dtype="float64")
    while True:
        pt_new = pt_old - lr * gradient
        nx, ny, nz, nv = pt_new
        f_new = f(nx, ny, nz, nv)
       
        if np.linalg.norm(gradient) < tolerance:
            break

        if f_new < f_old:
            lr /= lr_decay
            x, y, z, v = pt_new    
            gradient = np.array([f_dx(x, y), f_dy(x, y, z), f_dz(y, z, v), f_dv(z, v)], dtype="float64")        
            pt_old = pt_new
            f_old = f_new
        if f_new >= f_old:
            lr *= lr_decay
    
    # ugly print but its just for LaTeX
    print(f"$x_start = ({pt0[0]:.6f},{pt0[1]:.6f}, \
          {pt0[2]:.6f},{pt0[3]:.6f}), \
          x_min = ({x:.6f},{y:.6f},{z:.6f},{v:.6f}), \
          f(x_min) = {f(x, y, z, v):.6f}$")
    return pt_new, f_new


runs = 5
vals = []
pts = []
for _ in range(runs):
    point = np.random.uniform(-10, 10, size=4)
    pt, fv = gradient_descent(f, f_dx, f_dy, f_dz, f_dv, point)
    pts.append(pt)
    vals.append(fv)

pts = np.array(pts)
print(np.mean(pts, axis=0))
print(np.mean(vals))
