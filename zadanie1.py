import numpy as np

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


def pprint_results(solution, variables, alg_name):
    print(f'{alg_name} results:')
    for sol, var in zip(solution, variables):
        print(f'{var} = {sol}')
    print()


A = np.array([
    [3, 1, 0, 0, 0, 0, 0],
    [1, 4, 1, 0, 0, 0, 0],
    [0, 1, 4, 1, 0, 0, 0],
    [0, 0, 1, 4, 1, 0, 0],
    [0, 0, 0, 1, 4, 1, 0],
    [0, 0, 0, 0, 1, 4, 1],
    [0, 0, 0, 0, 0, 1, 3]
], dtype="float64")

u = np.zeros((A.shape[0], 1), dtype="float64")
v = np.zeros((A.shape[0], 1), dtype="float64")
u[0,0] = u[u.shape[0] - 1, 0] = v[0,0] = v[v.shape[0] - 1, 0] = 1.0
uvT = u @ v.T
B = A + uvT

A_constants = np.arange(start=1, stop=8, dtype="float64")
indepentent_variables = [f'x{n + 1}' for n in range(len(A_constants))]

np_results = np.linalg.solve(A, A_constants)
thomas_results = thomas_solve(A.copy(), A_constants.copy())

z = thomas_solve(A.copy(), A_constants.copy())
y = thomas_solve(A.copy(), u.reshape(u.shape[0]))

sherman_morisson_result = z - (v.T @ z) / (1 + v.T @ y) * y

np_matrix_b = np.linalg.solve(B, A_constants)

pprint_results(np_results, indepentent_variables, "np.linarg (matrix A)")
pprint_results(thomas_results, indepentent_variables, "thomas algorithm (matrix A)")
pprint_results(sherman_morisson_result, indepentent_variables, "sherman-morisson (matrix B)")
pprint_results(np_matrix_b, indepentent_variables, "np.linalg (matrix B)")
