
import numpy as np
from scipy.linalg import inv

def saddle_matrix_precondition(M, E, f, g):
    p, q = E.shape
    u = np.zeros((p, 1))
    lamb = np.zeros((q, 1))
    saddle_matrix = np.block([
        [M, E.conj().T],
        [E, np.zeros((q, q))]
    ])
    rhs = np.vstack([f, g])
    return saddle_matrix, rhs

def gsts_like_algorithm(B_c, B_1, E, omega_1, omega_2):
    B_u1_u2 = np.block([
        [B_1, omega_2 * E.conj().T],
        [-omega_1 * E, B_c - omega_1 * omega_2 * inv(E @ E.conj().T)]
    ])
    return B_u1_u2
