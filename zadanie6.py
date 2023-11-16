import numpy as np
from scipy import linalg as la

iterations = 1000
precision = 1e-10

np.set_printoptions(suppress=True,linewidth=150)

A = np.array([
    [2, -1, 0, 0, 1],
    [-1, 2, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 2, -1],
    [1, 0, 0, -1, 2]
    ], dtype="float64")

tau = 0.38197
n = A.shape[0]
A_res = A - tau * np.eye(n)

print(A_res)
permutation, L, U = la.lu(A_res, p_indices=True)

v0 = np.random.random(A.shape[0]).reshape(A.shape[0], 1)
v0 /= np.linalg.norm(v0)

for k in range(iterations):
    z = la.solve_triangular(L, v0[permutation, :], lower=True)
    v_new = la.solve_triangular(U, z)
    v_new /= np.linalg.norm(v_new)

    if (np.linalg.norm(np.abs(v0 - v_new)) < precision or
        np.allclose(np.abs(v0), np.abs(v_new), atol=precision)):
        print(f"Vector for eigenvalue: ~{tau}:")
        print(v_new)
        break
    
    v0 = v_new
