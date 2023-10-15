import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt

def precondition_system(A, b, B):
    """Apply preconditioning to a system of equations."""
    return np.dot(inv(B), A), np.dot(inv(B), b)

def decompose_matrix(A):
    """Decompose matrix into Hermitian and skew-Hermitian parts."""
    A_0 = 0.5 * (A + A.conj().T)
    A_1 = 0.5 * (A - A.conj().T)
    return A_0, A_1

def pttcm_method(A, b, B_e, omega_1, omega_2, max_iter=1000, tol=1e-8):
    """PTTCM method for solving a system of equations."""
    n = A.shape[0]
    v = np.zeros((n, 1))
    B_omega = inv(B_e + 0.5 * omega_1 * A) @ (B_e + 0.5 * omega_2 * A.conj().T)

    for _ in range(max_iter):
        v_new = B_omega @ v + omega_1 * inv(B_e) @ b
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new

    return v

def saddle_matrix_precondition(M, E, f, g):
    """Construct the saddle matrix preconditioned system."""
    p, q = E.shape
    u = np.zeros((p, 1))
    lamb = np.zeros((q, 1))
    saddle_matrix = np.block([
        [M, E.conj().T],
        [E, np.zeros((q, q))]
    ])
    rhs = np.vstack([f, g])
    return saddle_matrix, rhs

def gsts_method(M, E, B_1, B_2, f, g, omega_1, omega_2, gamma, max_iter=1000, tol=1e-8):
    """GSTS method for solving a system."""
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

def gsts_like_algorithm(B_c, B_1, E, omega_1, omega_2):
    """Define the GSTS-like algorithm."""
    B_u1_u2 = np.block([
        [B_1, omega_2 * E.conj().T],
        [-omega_1 * E, B_c - omega_1 * omega_2 * inv(E @ E.conj().T)]
    ])
    return B_u1_u2

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


def visualize_saddle_gradient_3D(u_value, lamb_value):
    """Visualize the gradient of the saddle point function in 3D."""
    
    # Use real part of the solution
    u_real = np.real(u_value)
    lamb_real = np.real(lamb_value)
    
    # Create a meshgrid around the solution (u, lambda)
    x = np.linspace(u_real - 2, u_real + 2, 20)
    y = np.linspace(lamb_real - 2, lamb_real + 2, 20)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2  # Saddle point function
    
    U = 2 * X  # Gradient in x-direction
    V = -2 * Y  # Gradient in y-direction
    W = np.zeros_like(X)  # No vertical component for gradient
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True, color="blue")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.5)
    ax.scatter(u_real, lamb_real, u_real**2 - lamb_real**2, c='r', s=100)  # Red dot represents our solution
    ax.set_title("3D Gradient Visualization of Saddle Point")
    ax.set_xlabel("u")
    ax.set_ylabel("lambda")
    ax.set_zlabel("Value")
    plt.show()

# Test visualization
visualize_saddle_gradient_3D(u[0], lamb[0])

# Additional 3D plot of the function f(x,y) = x^2 - y^2
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
x, y = np.meshgrid(x, y)
z = x**2 - y**2

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Plot of the function $f(x,y) = x^2 - y^2$')
plt.show()
