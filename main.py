
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def make_A_star(A):
    # Conjugate transpose is simply the transpose of the conjugate of A
    return np.conjugate(A).T

def extract_K_matrices(A1):
    # Initialize K_UP as a strictly upper triangular matrix from A1
    K_UP = np.triu(A1, k=1)
    
    # K_DOWN is the negative conjugate transpose of K_UP
    K_DOWN = -np.conjugate(K_UP).T
    
    return K_UP, K_DOWN
# Function definition for G(ω, τ) as provided
def G(omega, tau, A, B_c, K_L, K_U):
    """
    Computes the G(ω, τ) matrix.

    Parameters:
    omega (float): A positive parameter.
    tau (float): A positive parameter.
    A (np.ndarray): The matrix A.
    B_c (np.ndarray): The matrix B_c.
    K_L (np.ndarray): The strictly lower triangular matrix K_L.
    K_U (np.ndarray): The strictly upper triangular matrix K_U.

    Returns:
    np.ndarray: The G(ω, τ) matrix.
    """
    B_omega = (B_c + omega/2 * K_L) @ np.linalg.inv(B_c) @ (B_c + omega/2 * K_U)
    G_wt = np.linalg.inv(B_omega - tau * A)
    return G_wt


def solution_accuracy(A, b, solution):
    """
    Calculate the accuracy of a solution for the linear system Ax = b.

    Parameters:
    A (ndarray): Coefficient matrix.
    b (ndarray): Right-hand side vector.
    solution (ndarray): Solution vector of the system.

    Returns:
    float: The norm of the difference between b and Ax, representing the accuracy.
    """
    residual = b - np.dot(A, solution)
    return np.linalg.norm(residual)

# Updated iterative solution function to incorporate the provided G function
def iterative_solution(A,B_c,G, B, b, omega, tau, initial_v, H_0, tolerance=1e-7, max_iterations=1000):
    """
    Computes the iterative solution using the given iterative method.

    Parameters:
    G (callable): A function that takes omega, tau, A, B_c, K_L, K_U and returns the G matrix.
    B (callable): A function that takes omega, B_c, K_L, K_U and returns the B matrix.
    b (np.ndarray): The constant vector b.
    omega (float): A positive parameter.
    tau (float): A positive parameter.
    initial_v (np.ndarray): The initial approximation of the solution.
    tolerance (float): The tolerance for stopping the iterations.
    max_iterations (int): The maximum number of iterations.

    Returns:
    np.ndarray: The approximate solution vector.
    """
    A_star = make_A_star(A)
    A0 = 0.5*(A + A_star)
    A1 = 0.5*(A - A_star)

    K_U,K_L = extract_K_matrices(A1)

    v = initial_v
    for iteration in range(max_iterations):
        # Compute the next approximation
        G_wt = G(omega, tau, A, B_c, K_L, K_U)
        B_omega = B(B_c, K_L, K_U, H_0, omega)
        v_next = G_wt @ v + tau * np.linalg.inv(B_omega) @ b

        # Check for convergence
        #if np.linalg.norm(v_next - v) < tolerance:
        if solution_accuracy(A, b, v_next):
            #print(f"Converged in {iteration} iterations.")
            return v_next

        v = v_next

    #print(f"Reached maximum iterations without convergence.")
    return v

def B(omega, B_c, K_L, K_U):
    """
    Computes the B(ω) matrix.

    Parameters:
    omega (float): A positive parameter.
    B_c (np.ndarray): The matrix B_c.
    K_L (np.ndarray): The strictly lower triangular matrix K_L.
    K_U (np.ndarray): The strictly upper triangular matrix K_U.

    Returns:
    np.ndarray: The B(ω) matrix.
    """
    return (B_c + omega/2 * K_L) @ np.linalg.inv(B_c) @ (B_c + omega/2 * K_U)

def compute_B_omega(B_c, K_L, K_U, H_0, omega):
    """
    Computes the B(ω) matrix for the two-step skew-Hermitian iterative method.

    Parameters:
    B_c (np.ndarray): The Hermitian positive definite matrix B_c.
    K_L (np.ndarray): The strictly lower triangular part of matrix A.
    K_U (np.ndarray): The strictly upper triangular part of matrix A.
    H_0 (np.ndarray): The Hermitian matrix H_0.
    omega (float): A positive parameter.

    Returns:
    np.ndarray: The B(ω) matrix.
    """
    K_L_hat = K_L + H_0
    K_U_hat = K_U - H_0

    B_omega = (B_c + omega/2 * K_L_hat) @ np.linalg.inv(B_c) @ (B_c + omega/2 * K_U_hat)
    return B_omega

# #А должна быть положительно определена
def is_positive_definite(A):
    """
    Check if a matrix is positive definite.

    Parameters:
    A (np.ndarray): A square matrix.

    Returns:
    bool: True if the matrix is positive definite, False otherwise.
    """
    # Calculate the eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    # Check if all eigenvalues are greater than zero
    return np.all(eigenvalues > 0)

# #В данной работе предлолагается что diag(A1)=0
def check_diagonal_zeros(A):
    # np.diag returns the main diagonal of the matrix A.
    # np.all checks if all values are True (in this case, zeros).
    return np.all(np.diag(A) == 0)

# Определение матрицы системы и вектора свободных членов
A = np.array([[1, 1], [-1, 1]], dtype=complex)
#A = np.array([[2, 1+1j], [1-1j, 5]], dtype=complex)
print(is_positive_definite(A))
A_star = make_A_star(A)
A0 = 0.5*(A + A_star)
A1 = 0.5*(A - A_star)
print(check_diagonal_zeros(A1))
b = np.array([2, 0], dtype=complex)
# Начальное приближение
initial_v = np.array([0, 0], dtype=complex)
# Поскольку A уже является вещественной матрицей, B_c может быть просто единичной матрицей
B_c = np.eye(2, dtype=complex)#+A0+A
H_0 = np.zeros((2, 2), dtype=complex)#+A1
CONST_1 = 4
CONST_2 = 4
INTER = 100

# Определение функции для создания 3D графика точности решения с отображением оптимальной точки
def create_accuracy_3d_plot_with_optimal_point(A, B_c, G, B, b, initial_v, H_0, CONST_1, CONST_2, INTER):
    omega_values = np.linspace(-CONST_1, CONST_1, INTER)
    tau_values = np.linspace(-CONST_2, CONST_2, INTER)

    X, Y = np.meshgrid(omega_values, tau_values)
    Z = np.zeros(X.shape)

    min_residual_norm = np.inf
    optimal_omega = None
    optimal_tau = None
    optimal_v = None

    for i in tqdm(range(len(omega_values))):
        for j in range(len(tau_values)):
            try:
                v = iterative_solution(A, B_c, G, B, b, omega_values[i], tau_values[j], initial_v, H_0)
                residual_norm = solution_accuracy(A, b, v)

                Z[i, j] = residual_norm
                if residual_norm < min_residual_norm:
                    min_residual_norm = residual_norm
                    optimal_omega = omega_values[i]
                    optimal_tau = tau_values[j]
                    optimal_v = v

            except np.linalg.LinAlgError:
                Z[i, j] = np.nan

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    if optimal_omega is not None and optimal_tau is not None:

        # Now use these indices to get the correct Z-value
        ax.scatter(optimal_tau, optimal_omega, min_residual_norm, color='r', s=50)
        #min_residual_norm

    ax.set_xlabel('Omega')
    ax.set_ylabel('Tau')
    ax.set_zlabel('Solution Accuracy')
    ax.set_title('Solution Accuracy with Optimal Point')

    print(np.nanmin(Z))

    return (optimal_omega, optimal_tau), optimal_v, fig

# Example usage
(optimal_parameters, optimal_v, fig) = create_accuracy_3d_plot_with_optimal_point(A, B_c, G, compute_B_omega, b, initial_v, H_0, CONST_1, CONST_2, INTER)


# Optimal parameters and sol

#Run the parameter search
print(f"Optimal parameters (omega, tau): {optimal_parameters}")
print(f"Solution with optimal parameters: {optimal_v}")
print(f"Solution accuracy: {solution_accuracy(A, b, optimal_v)}")
#Проверка на адекватность
solution_matrix = np.linalg.solve(A, b)
print(f"Real solituion: {solution_matrix}")

# Optimal parameters
print(f"Optimal parameters (omega, tau): {optimal_parameters}")

plt.show()