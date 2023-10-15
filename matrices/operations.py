
import numpy as np
from scipy.linalg import inv

def decompose_matrix(A):
    A_0 = 0.5 * (A + A.conj().T)
    A_1 = 0.5 * (A - A.conj().T)
    return A_0, A_1

def precondition_system(A, b, B):
    return np.dot(inv(B), A), np.dot(inv(B), b)
