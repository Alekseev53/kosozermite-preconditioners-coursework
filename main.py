
import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
from iterative_methods.gsts import gsts_method, pttcm_method
from matrices.operations import decompose_matrix, precondition_system
from matrices.preconditioners import saddle_matrix_precondition, gsts_like_algorithm

# Mock data for testing purposes
n = 3
A = np.array([
    [2, 1, 0],
    [1, 3, 1],
    [0, 1, 4]
], dtype=complex)

b = np.array([
    [1],
    [2],
    [3]
], dtype=complex)

B = np.eye(n, dtype=complex)

M = np.array([
    [2, 0],
    [0, 3]
], dtype=complex)

E = np.array([
    [1, 0],
    [0, 1]
], dtype=complex)

f = np.array([
    [1],
    [2]
], dtype=complex)

g = np.array([
    [1],
    [2]
], dtype=complex)

B_1 = np.eye(2, dtype=complex)
B_2 = np.eye(2, dtype=complex)

omega_1 = 0.5
omega_2 = 0.5
gamma = 1.0

# Test functions
preconditioned_A, preconditioned_b = precondition_system(A, b, B)
A_0, A_1 = decompose_matrix(A)
v = pttcm_method(A, b, B, omega_1, omega_2)
saddle_matrix, rhs = saddle_matrix_precondition(M, E, f, g)
u, lamb = gsts_method(M, E, B_1, B_2, f, g, omega_1, omega_2, gamma)
B_u1_u2 = gsts_like_algorithm(B_2, B_1, E, omega_1, omega_2)

print(preconditioned_A)
print(preconditioned_b)
print(A_0)
print(A_1)
print(v)
print(saddle_matrix)
print(rhs)
print(u)
print(lamb)
print(B_u1_u2)
