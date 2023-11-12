import numpy as np

iterations = 1000
precision = 1e-10

A = np.array([
    [19, 13, 10, 10, 13, -17],
    [13, 13, 10, 10, -11, 13],
    [10, 10, 10, -2, 10, 10],
    [10, 10, -2, 10, 10, 10],
    [13, -11, 10, 10, 13, 13],
    [-17, 13, 10, 10, 13, 19]
], dtype="float32")

A /= 12

# choose a random vector
v0 = np.random.random(A.shape[0]).reshape(A.shape[0], 1)
v0 /= np.linalg.norm(v0)
v_found = None
for k in range(iterations):
    v_new = A @ v0
    if v_found is not None:
        v_new = v_new - v_found * (v_found.T @ v_new)

    v_new = v_new / np.linalg.norm(v_new) 

    if np.linalg.norm(np.abs(v_new - v0)) < precision:
        eigenval = (v_new.T @ A @ v_new)
        print(v_new, eigenval)
        if v_found is not None:
            break
        v_found = v_new

    v0 = v_new
