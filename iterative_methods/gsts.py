
import numpy as np
from scipy.linalg import inv

def gsts_method(M, E, B_1, B_2, f, g, omega_1, omega_2, gamma, max_iter=1000, tol=1e-8):
    p, q = E.shape
    w = np.zeros((p + q, 1))
    for _ in range(max_iter):
        u = w[:p]
        mu = w[p:]
        u_new = B_2 @ (u + omega_1 * (f - M @ u - E.conj().T @ mu + E @ u - g))
        mu_new = B_1 @ (mu - omega_2 * M @ u_new + omega_2 * E.conj().T @ u_new - gamma * mu - u_new + f)
        w_new = np.vstack([u_new, mu_new])
        if np.linalg.norm(w_new - w) < tol:
            break
        w = w_new

    return w[:p], w[p:]

def pttcm_method(A, b, B_e, omega_1, omega_2, max_iter=1000, tol=1e-8):
    n = A.shape[0]
    v = np.zeros((n, 1))
    B_omega = inv(B_e + 0.5 * omega_1 * A) @ (B_e + 0.5 * omega_2 * A.conj().T)

    for _ in range(max_iter):
        v_new = B_omega @ v + omega_1 * inv(B_e) @ b
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new

    return v
