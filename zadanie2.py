import numpy as np
import matplotlib.pyplot as plt
# poprawic z uwzglednieniem struktury macierzy

N = 128
A = np.zeros((N,N), dtype="float32")
iterations = 1000
precision = 1e-10

for i in range(len(A) - 1):
    A[i, i]  = 4.0
    A[i, i+1] = 1.0
    A[i+1, i] = 1.0

A[N-1, N-1] = 4.0

for i in range(len(A)- 4):
    A[i, i+4] = 1.0
    A[i+4, i] = 1.0

A_constants = np.ones(N, dtype="float32")
np_sol = np.linalg.solve(A, A_constants)

def gauss_seidel(A, b):
    x0 = np.zeros(N, dtype="float32")
    norms = []
    for k in range(iterations):
        x_old = x0.copy()
        for i in range(N):
            sigma = 0.0
            for j in range(N):
                if i != j:
                    sigma += A[i,j] * x0[j]
            x0[i] = (b[i] - sigma) / A[i, i]
        norm = np.linalg.norm(np.abs(x0 - x_old))
        norms.append(norm)
        
        if norm < precision:
            norms = np.array(norms)
            norms_closeup = norms[norms < 1e-5]
            ks = np.arange(k+1)
            ks_closeup = np.arange(len(norms) - len(norms_closeup), k+1)
            ks_closeup = ks_closeup.reshape(len(ks_closeup))
            return x0, norms, ks, norms_closeup, ks_closeup


def conjugate_gradient(A, b):
    x0 = np.zeros(N, dtype="float32")
    norms = []
    r = b.copy()
    p  = r.copy()
    for k in range(iterations):
        alpha = np.dot(r, r) / np.dot(p, A @ p)
        x0 = x0 + alpha * p
        r_next = r - alpha * A @ p
        norm = np.linalg.norm(r_next)
        norms.append(norm)

        if norm < precision:
            norms = np.array(norms)
            norms_closeup = norms[norms < 1e-5]
            ks = np.arange(k+1)
            ks_closeup = np.arange(len(norms) - len(norms_closeup), k+1)
            ks_closeup = ks_closeup.reshape(len(ks_closeup))
            return x0, norms, ks, norms_closeup, ks_closeup

        beta = np.dot(r_next, r_next) / np.dot(r,r)
        p = r_next + beta * p
        r = r_next

sol1, norms1, ks1, norms_closeup1, ks_closeup1 = gauss_seidel(A, A_constants)
sol2, norms2, ks2, norms_closeup2, ks_closeup2 = conjugate_gradient(A, A_constants)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.set_title("Gauss-Seidel vs Conjguate gradient norm convergence", fontdict={'fontsize': 10})
ax1.set_xlabel("iteration")
ax1.set_ylabel("norm value")
ax1.plot(ks1, norms1, label="Gauss-Seidel")
ax1.plot(ks2, norms2, label="Conjugate Gradient")
ax1.legend()
ax2.set_title(f"Gauss-Seidel vs Conjugate gradient norm convergence (for norms < 1e-5)", fontdict={'fontsize': 10})
ax2.set_xlabel("iteration")
ax2.set_ylabel("norm value")
ax2.plot(ks_closeup1, norms_closeup1, label="Gauss-Seidel")
ax2.plot(ks_closeup2, norms_closeup2, label="Conjugate Gradient")
ax2.legend()
plt.savefig("gauss-seidel_vs_conjugate-gradient.png",dpi=500)

print("Gauss-Seidel solution")
print(sol1)
print("np.allclose comparision to np.linalg.solve: ", np.allclose(np_sol, sol1))
print("Conjugate gradient solution")
print(sol2)
print("np.allclose comparision to np.linalg.solve: ", np.allclose(np_sol, sol2))
print("np.allclose comparision between solutions: ", np.allclose(sol1, sol2))






