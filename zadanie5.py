import numpy as np

np.set_printoptions(suppress=True, linewidth=150)
A = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype="float32")

B = np.array([
    [0, 0, 0, -1],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0]
], dtype="float32")

C = np.array([
    [0,1,0, -1j],
    [1,0,-1j,0],
    [0,1j,0,1],
    [1j,0,1,0]
], dtype=complex)

H = np.block([[A, -B], [B, A]])

print(H)

def qr_francis(A, num_iterations=100000, tol=1e-5, shift=1e-1):
    n = A.shape[0]
    eigenvectors = np.eye(n)
    Ac = A.copy()
    shift_matrix = shift * np.eye(n) 

    for _ in range(num_iterations):
        Q, R = np.linalg.qr(Ac - shift_matrix)
        Ac = R @ Q + shift_matrix
        eigenvectors = eigenvectors @ Q

        if np.allclose(Ac, np.diag(np.diagonal(Ac)), atol=tol):
            break

    eigenvalues = np.diagonal(Ac)
    return eigenvalues, eigenvectors

evals_H, evecs_H = qr_francis((H))
print(evals_H, evecs_H)

# cast for indexing
N = int(H.shape[0] / 2)
# real part
x = evecs_H[:N ,:]
# imaginary part multiplied by i
y = evecs_H[N:, :] * 1j

eigenvectors_HC = x + y
eigenvectors_C = np.zeros((N, N), dtype=complex)

unique_eigenvalues = np.unique(evals_H)
for i in range(len(unique_eigenvalues)):
    current_eval = unique_eigenvalues[i]
    # find the first occurence of eigenvalue
    eigenvector_idx = np.argmax(evals_H == current_eval)
    # take its corresponding vector
    eigenvectors_C[:, i] = eigenvectors_HC[:, eigenvector_idx]

# orthogonality check
for i in range(N):
    for j in range(N):
        if i != j:
            print(np.vdot(eigenvectors_C[:, i], eigenvectors_C[:, j]))

print("My solution")
print(unique_eigenvalues)
print(eigenvectors_C)

eigenvalues_C, eigenvectors_C = np.linalg.eigh(C)

print("-----")
print("numpy solution")
print(eigenvalues_C)
print(eigenvectors_C)