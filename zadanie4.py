import numpy as np

np.set_printoptions(suppress=True)

A = np.array([
    [19, 13, 10, 10, 13, -17],
    [13, 13, 10, 10, -11, 13],
    [10, 10, 10, -2, 10, 10],
    [10, 10, -2, 10, 10, 10],
    [13, -11, 10, 10, 13, 13],
    [-17, 13, 10, 10, 13, 19]
], dtype="float32")

A /= 12

def qr_algorithm(A, num_iterations=1000, tolerance=1e-3):
    n = A.shape[0]
    eigenvectors = np.eye(n)
    Q, R = qr_decomposition(A)
    Q_old = np.empty(Q.shape)

    for i in range(num_iterations):
        Q_old[:] = Q
        tridiagonal_matrix = A @ Q
        eigenvectors = eigenvectors @ Q
        Q, R = qr_decomposition(tridiagonal_matrix)

        if np.allclose(Q, Q_old, atol=tolerance):
            print("iterations:", i)
            break

    eigenvalues = R.diagonal()
    return eigenvalues


def householder_tridiag(A):
    n = A.shape[0]
    Ac = A.copy()
    for i in range(n - 1):
        alpha = -np.sign(Ac[i+1,i]) * np.linalg.norm(Ac[i+1:,i])
        r = np.sqrt(0.5 * (alpha * alpha - Ac[i+1,i] * alpha))
        v = np.zeros_like(Ac[i:, i]).reshape(-1, 1)
        v[1, 0] = (Ac[i+1,i] - alpha) / (2 * r)
        v[2:, 0] = Ac[i+2:, i] / (2 * r)
        I = np.eye(n-i)
        P = I - 2 * (v @ np.transpose(v))
        Ac[i:, i:] = P @ Ac[i:, i:] @ P
    return Ac


def qr_decomposition(A):
    n = A.shape[0]
    R = A.copy()
    Q = np.eye(n)
    for i in range(n - 1):
        alpha = -np.sign(R[i,i]) * np.linalg.norm(R[i:,i])
        r = np.sqrt(0.5 * (alpha * alpha - R[i,i] * alpha))
        v = np.zeros_like(R[i:, i]).reshape(-1, 1)
        v[0, 0] = (R[i, i] - alpha) / (2 * r)
        v[1:, 0] = R[i+1:, i] / (2 * r)
        P = np.eye(n-i) - 2 * (v @ np.transpose(v))
        R[i:, i:] = P @ R[i:, i:]
        Q[:, i:] = Q[:, i:] @ P
    return Q, R


A_tri = householder_tridiag(A)
eigenvalues = qr_algorithm(A_tri)
print(A_tri)
print(eigenvalues)
