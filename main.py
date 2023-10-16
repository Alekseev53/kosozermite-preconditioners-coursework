from scipy.linalg import null_space
#\section*{Введение}
#12 - 26

#Пояснения
"""
Давайте последовательно разберёмся в каждом из вопросов:
1. **Что такое предобусловленность?**
   Предобусловленность — это процедура, применяемая к системе 
   уравнений для улучшения её численных свойств. Суть метода 
   заключается в умножении исходной системы уравнений на так 
   называемый "предобусловитель" с целью сделать новую систему 
   более "хорошо обусловленной" или, другими словами, 
   уменьшить число обусловленности матрицы системы. 
   В результате, итерационные методы могут сходиться быстрее 
   к решению новой системы, чем к решению исходной.
2. **Что такое седловая задача?**
   Седловые задачи — это задачи оптимизации, где функционал 
   имеет как минимумы, так и максимумы в различных областях 
   пространства решений. Графически такая функция напоминает 
   седло коня, отсюда и название. В контексте систем линейных
     уравнений седловая задача часто связана с системами, 
     имеющими матрицу, которая не является ни положительно 
     определенной, ни отрицательно определенной.
3. **Что считается "большой размерностью"?**
   "Большая размерность" — 10^4 10^6 10^9 уравнений
4. **Что такое подпространства Крылова?**
   Подпространство Крылова — это последовательность векторных 
   пространств, генерируемых векторами вида 
   \( b, Ab, A^2b, \ldots, A^kb \), 
   где \( A \) — матрица, 
   а \( b \) — вектор. 
   Эти пространства играют ключевую роль в итерационных 
   методах для решения систем линейных уравнений, таких 
   как методы Крылова.
5. **Что за метод GMRES?**
   GMRES (Generalized Minimal RESidual) — это итерационный 
   метод решения систем линейных уравнений. Он основан на 
   поиске решения в подпространствах Крылова таким образом, 
   чтобы минимизировать норму невязки. GMRES является одним 
   из наиболее популярных методов решения несимм
"""

"""
\[ Av = b, \quad b \in \mathbb{C}^n, \]
где \( A \in \mathbb{C}^{n \times n} \) — 
нерегулярно положительно определенная матрица.
"""

#Пояснения
"""
Для проверки того, является ли матрица положительно 
определенной, можно использовать различные критерии.
 Одним из наиболее распространенных методов является 
 проверка собственных значений матрицы: матрица является 
 положительно определенной тогда и только тогда, когда 
 все её собственные значения строго положительны.
"""
import numpy as np

def is_positive_definite(A):
    eigenvalues = np.linalg.eigvals(A)
    return np.all(eigenvalues > 0)

# Размерность матрицы и вектора
n = 10**2

"""
\[ Av = b, \quad b \in \mathbb{C}^n, \]
"""

# Генерация случайной матрицы
# Создаем функцию для генерации случайной положительно определенной матрицы
def generate_positive_definite_matrix(n):
    # Генерация случайной матрицы
    M = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    # Преобразование матрицы в положительно определенную
    return M @ np.conj(M.T)

# Генерация положительно определенной матрицы
A = generate_positive_definite_matrix(n)

# Проверка
is_pos_def = is_positive_definite(A)

# Возвращаем результат проверки и первые несколько элементов для демонстрации
A_sample = A[:5, :5]
print(is_pos_def)

# Генерация случайного вектора
b = np.random.randn(n) + 1j * np.random.randn(n)

# Возвращаем первые несколько элементов для демонстрации
b_sample = b[:5]

print(A_sample, b_sample)

#24

def condition_number(matrix):
    """Вычисляет число обусловленности матрицы."""
    return np.linalg.cond(matrix)

def improve_conditioning(A, B):
    """Проверяет, является ли матрица B^-1*A лучше обусловленной, чем A."""
    original_cond = condition_number(A)
    preconditioned_cond = condition_number(np.linalg.inv(B) @ A)
    
    return original_cond, preconditioned_cond, preconditioned_cond < original_cond

# Возьмем пример с меньшей размерностью из-за ограничений по памяти
n_small = 500

# Генерация случайной матрицы A и предобусловителя B
A_small = np.random.randn(n_small, n_small) + 1j * np.random.randn(n_small, n_small)
B_small = generate_positive_definite_matrix(n_small)

original_cond, preconditioned_cond, is_better = improve_conditioning(A_small, B_small)

print(original_cond, preconditioned_cond, is_better)

#B^-1*A
#имеет наибольшое число собственных значений на периферии своего спекта

# Вычисление собственных значений матрицы B^-1*A
eigenvalues = np.linalg.eigvals(np.linalg.inv(B_small) @ A_small)

# Определение радиуса спектра (расстояния от начала координат до наиболее удаленного собственного значения)
spectral_radius = np.max(np.abs(eigenvalues))

# Определение периферийных собственных значений (те, которые близки к радиусу спектра)
tolerance = 0.01  # допустимая погрешность для определения "близости"
peripheral_eigenvalues = eigenvalues[np.abs(np.abs(eigenvalues) - spectral_radius) < tolerance]

print(spectral_radius, peripheral_eigenvalues)

#Второй лист
#\section{Итерационные методы решения СЛАУ с сильно нерегулярной матрицей}

"""
Матрицу \(A\) системы (1) представим в виде суммы её 
диагонали и косоэрмитовой части:
\[ A = A_0 + A_1, \]
где
\[ A_0 = \frac{1}{2}(A + A^*), 
\quad A_1 = \frac{1}{2}(A - A^*). \] (3)

Что такое косоэрмитовая часть ?
Что такое A^*
"""

#Пояснения
"""
1. **Косоэрмитова часть матрицы**:
   Косоэрмитова (или антиэрмитова) часть матрицы — это 
   часть матрицы, которая остается после вычитания её 
   эрмитовой (или сопряженно-транспонированной) части.
   Другими словами, это часть матрицы, которая "отвечает" 
   за асимметричность в отношении её эрмитовой части.

2. **\(A^*\)**:
   Символ \(A^*\) обозначает эрмитово сопряжение матрицы 
   \(A\). Это комбинация транспонирования матрицы (замена 
   строк на столбцы) и взятия комплексного сопряжения каждого 
   элемента. Для реальных матриц эрмитово сопряжение 
   эквивалентно обычному транспонированию.

В формулах, которые вы предоставили:
- \(A_0\) представляет собой эрмитову часть матрицы \(A\), 
которая является симметричной относительно своей главной 
диагонали.
- \(A_1\) представляет собой косоэрмитову часть матрицы \(A\), 
которая является асимметричной относительно своей главной 
диагонали.
"""

def decompose_matrix(A):
    """
    Декомпозиция матрицы A на её диагональную и косообразную части.
    """
    A_star = np.conj(A.T)  # Эрмитово сопряжение
    A_0 = 0.5 * (A + A_star)
    A_1 = 0.5 * (A - A_star)
    
    return A_0, A_1

def is_positive_definite_diag_part(A_0):
    """
    Проверяет, является ли диагональная часть матрицы A положительно определенной.
    """
    eigenvalues = np.linalg.eigvals(A_0)
    return np.all(eigenvalues > 0)

def matrix_norm(matrix):
    """
    Возвращает матричную норму (спектральную норму).
    """
    return np.linalg.norm(matrix, ord=2)

def is_strongly_irregular(A_0, A_1):
    """
    Проверяет, является ли матрица A сильно нерегулярной.
    """
    norm_A_0 = matrix_norm(A_0)
    norm_A_1 = matrix_norm(A_1)
    
    return norm_A_0 < 0.1 * norm_A_1  # Просто пример порога для определения "намного меньше"

def check_diag_A1(A_1):
    """
    Проверяет, равна ли диагональ A_1 нулю.
    """
    return np.all(np.diag(A_1) == 0)

# Пример блочной матрицы 4x4
A_example = np.array([[1, 2, 1, 0],
                      [3, 4, 0, 1],
                      [0, 1, 5, 6],
                      [7, 8, 2, 3]])

# Применение функций к ранее использованной матрице A_example
A_0_example, A_1_example = decompose_matrix(A_example)
is_pos_def = is_positive_definite_diag_part(A_0_example)
is_irregular = is_strongly_irregular(A_0_example, A_1_example)
is_diag_zero = check_diag_A1(A_1_example)

print(is_pos_def, is_irregular, is_diag_zero)


r"""
35
Представим косообразную часть \( A_1 \) матрицы \( A \) в виде:
\[ A_1 = K_l + K_u, \] (4)

где \( K_l \) и \( K_u \) строго ниже- и верхнетреугольная матрица соответственно. Очевидно, что \( K_l = -K^*_u \).
"""

def decompose_skew_hermitian(A_1):
    """
    Декомпозиция косоэрмитовой матрицы A_1 на её строго нижне- и верхнетреугольные части.
    """
    # Строго верхнетреугольная часть
    K_u = np.triu(A_1, 1)
    
    # Строго нижнетреугольная часть (с учетом свойства K_l = -K^*_u)
    K_l = -np.conj(K_u.T)
    
    return K_l, K_u

# Применение функции к матрице A_1_example
K_l_example, K_u_example = decompose_skew_hermitian(A_1_example)

print(K_l_example[:5, :5], K_u_example[:5, :5])  # Возвращаем первые несколько элементов для демонстрации


r"""
Метод ПТТКМ [5]. Пусть задано начальное приближение \( v(0) \) и положительные параметры \( h \) и \( \tau \). Для \( p = 0,1, ... \) достаточная последовательность приближений \( \{v(p)\} \) вычисляется:

\[ v(p+1) = G(v, \tau)v(p) + \tau B(\omega)^{-1}b, \]
где \( G(v, \tau) = B(\omega)^{-1}(B(\omega) - \tau A) \), \( B(\omega) \in \mathbb{C}^{n \times n} \) определяется следующим образом:
\[ B(\omega) = (B_e + \frac{\omega}{2} K_l)^{-1}(B_e + \frac{\omega}{2} K_u). \]

Здесь, \( B_e \in \mathbb{C}^{n \times n} \) — симметричная положительно определенная матрица.

\[ B(\omega) \] исследован для двух параметрических методов, для которого:
\[ B(\omega_1, \omega_2) = (B_e + \omega_1 K_l)^{-1}(B_e + \omega_2 K_u), \] (5)

где \( \omega_1 \) и \( \omega_2 \) —  неотрицательные параметры, не равные нулю одновременно
"""
def B_omega(omega, B_e, K_l, K_u):
    """
    Вычисление матрицы B(ω) по заданной формуле.
    """
    return np.linalg.inv(B_e + (omega / 2) * K_l) @ (B_e + (omega / 2) * K_u)

def G(v, tau, B_omega, A):
    """
    Вычисление матрицы G(v, τ) по заданной формуле.
    """
    return np.linalg.inv(B_omega) @ (B_omega - tau * A)

def PTTKM_method(v0, h, tau, A, b, B_e, K_l, K_u, max_iter=100):
    """
    Применение метода ПТТКМ для вычисления последовательности приближений.
    """
    v = [v0]
    for p in range(max_iter):
        B_current = B_omega(h, B_e, K_l, K_u)
        G_current = G(v[p], tau, B_current, A)
        v_next = G_current @ v[p] + tau * np.linalg.inv(B_current) @ b
        v.append(v_next)
    return v

def B_omega_parametric(omega1, omega2, B_e, K_l, K_u):
    """
    Вычисление матрицы B(ω1, ω2) по заданной формуле.
    
    Параметры:
    - omega1, omega2: неотрицательные параметры, не равные нулю одновременно.
    - B_e: симметричная положительно определенная матрица.
    - K_l, K_u: матрицы, соответствующие определению.
    
    Возвращает:
    - Матрицу B(ω1, ω2)
    """
    # Проверка параметров omega1 и omega2
    if omega1 < 0 or omega2 < 0:
        raise ValueError("Both omega1 and omega2 should be non-negative.")
    if omega1 == 0 and omega2 == 0:
        raise ValueError("Both omega1 and omega2 cannot be zero simultaneously.")
    
    return np.linalg.inv(B_e + omega1 * K_l) @ (B_e + omega2 * K_u)



# Простой пример использования
# Начальные параметры
n = 4
v0 = np.array([1, 1, 1, 1])
h = 0.1
tau = 0.01
B_e_example = np.eye(n)  # Просто единичная матрица в качестве примера
# Простой тест функции с проверкой
try:
    B_omega_parametric(-1, 1, B_e_example, np.eye(n), np.eye(n))
except ValueError as e:
    test_output = str(e)

# Применение метода
v_sequence = PTTKM_method(v0, h, tau, A_example, \
                          np.array([1, 2, 3, 4]), B_e_example, \
                          K_l_example, K_u_example)

# Возвращаем первые 5 элементов последовательности для демонстрации
print(v_sequence[:5])

#Третий лист

"""
% Insert the provided content below
В [7, 8] предложен и двухшаговый координатный итерационный метод (ДКМ), даны достаточные условия сходимости метода и выбор оптимальных итерационных параметров. Матрица \( B(\omega) \) для ДКМ имеет вид
\[ B(\omega) = \left(B_c + \frac{\omega}{2} K_L\right)B_E \left(B_c + \frac{\omega}{2} K_U\right), \]
где \( K_L = K_L + H_0, K_U = K_U - H_0 \). Но \( B_c = C_{\times} \) – некоторая разряженная матрица, \( B_E = C_{\times} \) – зрящая по локально квазиточно определенная матрица. Очевидно, что \( K_L = -K_U; A = (K_L + H_0) + (K_U - H_0) = K_L + K_U \). 
В случае, когда \( H_0 = 0 \), ДКМ сводится к ИТКМ, специальный выбор матрицы \( H_0 \) позволяет улучшить сходимость метода.
В [9] впервые предложен обобщенный разряженный треугольный метод GSTS (Generalized Skew-Hermitian Triangular Splitting) для решения основных СЛАУ, блочно-структурированных матриц которых имеет положительно определенный (1, 1) блок.
В данной работе исследуются свойства матрицы \( B(\omega) \) (6), используя в качестве предобусловленной матрицы матрицу СЛАУ (1). Исследованы задачи рассмотрены более общий случай, когда (1, 1) матричный блок близко кнонструктуре портфолио СЛАУ с такой матрицей используется метод расширенного Лагранжиана. Доказана теорема о распределении спектра матрицы \( B^{-1}(\omega_1, \omega_2) \). Для предобусловливателя \( B(\omega_1, \omega_2) \) есть обобщение \( B(\omega_1, \omega_2) \) (5) для основных задач.
"""

def B_omega_DKM(omega, B_c, B_e, K_L, K_U):
    """
    Вычисление матрицы B(ω) для ДКМ по заданной формуле.
    """
    return (B_c + (omega / 2) * K_L) @ B_e @ (B_c + (omega / 2) * K_U)

def update_K_matrices(K_L, H_0):
    """
    Обновление матриц K_L и K_U на основе матрицы H_0.
    """
    K_L_updated = K_L + H_0
    K_U_updated = -K_L_updated
    return K_L_updated, K_U_updated

def compute_A_from_K(K_L, K_U, H_0):
    """
    Вычисление матрицы A на основе матриц K_L и K_U.
    """
    return (K_L + H_0) + (K_U - H_0)

# Простые тесты
# Допустим, у нас есть следующие матрицы:
B_c_example = np.eye(n) * 2
B_e_example = np.eye(n)
K_L_example = np.eye(n)
H_0_example = np.zeros((n, n))

K_L_updated, K_U_updated = update_K_matrices(K_L_example, H_0_example)
A_example = compute_A_from_K(K_L_updated, K_U_updated, H_0_example)
B_omega_DKM_example = B_omega_DKM(0.5, B_c_example, B_e_example, K_L_updated, K_U_updated)

print(B_omega_DKM_example)


"""
\section*{Предобусловливание СЛАУ с седловой матрицей}

Характерно к задачей, приводящих к решению СЛАУ с седловой матрицей, является следующая задача квадратного программирования: необходимо найти минимум \( J(u) \) на утверждении \( J(u) = \frac{1}{2}u^* E u - u^* f \) при наличии \( q \leq p \) линейных ограничений \( E u = g \):
\[
\begin{aligned}
\left( \begin{array}{cc}
M & E^* \\
E & 0 
\end{array} \right)
\left( \begin{array}{c}
u \\
\lambda
\end{array} \right)
= 
\left( \begin{array}{c}
f \\
g
\end{array} \right),
\end{aligned}
\]
где \( M = M^* = C_{\times} \) – положительно определенная матрица, \( E \) ∈ \( C^{q \times p} \) – произвольная матрица полного ранга, \( q \leq p \), \( u \) ∈ \( C^p \), \( g \) ∈ \( C^q \). Данной задаче соответствует функционал Лагранжа \( L(u, \lambda) = J(u) + \mu^* (E u - g) \), где \( \mu \) – вектор Лагранжевых множителей. Заметим, что матрица блочно-структурованной СЛАУ (7) неотрицательна тогда и только тогда, когда [10]:
\[
\begin{aligned}
\text{rank}(E^*) = q, \\
\text{ker}(E) \cap \text{ker}(M) = \{0\}.
\end{aligned}
\]
"""

def saddle_point_system(M, E, f, g):
    """
    Формирование СЛАУ с седловой матрицей.
    """
    matrix = np.block([
        [M, E.T],
        [E, np.zeros((E.shape[0], E.shape[0]))]
    ])
    
    rhs = np.concatenate([f, g])
    
    return matrix, rhs

def lagrangian_functional(u, mu, E, f, g, M):
    """
    Функционал Лагранжа L(u, λ).
    """
    J = 0.5 * u.conj().T @ M @ u - u.conj().T @ f
    return J + mu.conj().T @ (E @ u - g)

def check_matrix_conditions(E, M):
    """
    Проверка условий на неотрицательность матрицы блочно-структурированной СЛАУ из [10].
    """
    rank_E_star = np.linalg.matrix_rank(E.T)
    ker_E = null_space(E)
    ker_M = null_space(M)
    
    intersection = np.isclose(ker_E, ker_M).all()
    
    return rank_E_star == E.shape[0] and not intersection

# Пример использования
M_example = np.array([[2, 0], [0, 3]])
E_example = np.array([[1, 1]])
f_example = np.array([1, 2])
g_example = np.array([1])

matrix, rhs = saddle_point_system(M_example, E_example, f_example, g_example)
L_example = lagrangian_functional(np.array([1, 1]), np.array([1]), E_example, f_example, g_example, M_example)
conditions_met = check_matrix_conditions(E_example, M_example)

print(matrix, rhs, L_example, conditions_met)



#Четвертый лист

r"""
\text{Преобразуем } (3) \text{ к эквивалентной неявной СЛАУ, матрица которой имеет сектр, дежавный и право и поу полоскости } [11]:
\begin{equation}
\begin{pmatrix}
M & E^* \\
-E & 0 \\
\end{pmatrix}
\begin{pmatrix}
u \\
\mu \\
\end{pmatrix}
=
\begin{pmatrix}
f \\
-g \\
\end{pmatrix}
. (8)
\end{equation}

\text{Рассмотрим случай, когда } (1,1) \text{ маргиналь блок полурегелен или вырожден. Будем использовать метод расширенно Лагранжиана, который состоит в замене } (8) \text{ на СЛАУ}
\begin{equation}
\Delta w =
\begin{pmatrix}
M & E^* \\
-E & 0 \\
\end{pmatrix}
\begin{pmatrix}
u \\
\mu \\
\end{pmatrix}
=
\begin{pmatrix}
f + \gamma E^*g \\
-g \\
\end{pmatrix}
= F, (9)
\end{equation}
\text{в которой } M \text{ заменяется матрицу } M = M + \gamma E^*E, \text{ являющуюся положительно определенной для всех } \gamma > 0, \text{ если } M \text{ имеет полный ранг. Очевидно, что } (9) \text{ имеет тоже самое решение, что и } (8). \text{Наиболее эффективный выбор } \gamma = ||M||_2/||E||_2 [12]. \text{В этом случае число обусловленности как } (1,1) \text{ блока, так и всей матрицы коэффициентов является наименьшим.}
\text{Представим матрицу } A \text{ из } (9), \text{ аналогично } (2), \text{ в виде суммы ее эрмитовой и коэрмитовой составляющих:}

A = A_0 + A_1, \quad A_0 = \begin{pmatrix} M & 0 \\ 0 & E^* \end{pmatrix}, \quad A_1 = \begin{pmatrix} 0 & E^* \\ -E & 0 \end{pmatrix}.
"""
def implicit_system(M, E, f, g):
    """
    Формирование неявной СЛАУ (8).
    """
    matrix = np.block([
        [M, E.T],
        [-E, np.zeros((E.shape[0], E.shape[0]))]
    ])
    
    rhs = np.concatenate([f, -g])
    
    return matrix, rhs

def extended_lagrangian_system(M, E, f, g, gamma):
    """
    Применение метода расширенного Лагранжиана для получения системы (9).
    """
    M_modified = M + gamma * E.T @ E
    rhs_modified = np.concatenate([f + gamma * E.T @ g, -g])
    
    matrix = np.block([
        [M_modified, E.T],
        [-E, np.zeros((E.shape[0], E.shape[0]))]
    ])
    
    return matrix, rhs_modified

def decompose_matrix_A(M, E):
    """
    Разделение матрицы A на её эрмитовую и коэрмитовую составляющие.
    """
    A0 = np.block([
        [M, np.zeros((M.shape[0], E.shape[0]))],
        [np.zeros((E.shape[0], M.shape[1])), np.zeros((E.shape[0], E.shape[0]))]
    ])
    
    A1 = np.block([
        [np.zeros_like(M), E.T],
        [-E, np.zeros((E.shape[0], E.shape[0]))]
    ])
    
    return A0, A1

# Простой тест
M_example = np.array([[2, 0], [0, 3]])
E_example = np.array([[1, 1]])
f_example = np.array([1, 2])
g_example = np.array([1])
gamma_example = np.linalg.norm(M_example, 2) / np.linalg.norm(E_example, 2)

matrix_8, rhs_8 = implicit_system(M_example, E_example, f_example, g_example)
matrix_9, rhs_9 = extended_lagrangian_system(M_example, E_example, f_example, g_example, gamma_example)
A0_example, A1_example = decompose_matrix_A(M_example, E_example)

print(matrix_8, rhs_8, matrix_9, rhs_9, A0_example, A1_example)



#Пятый лист

#Шестой лист
r"""
т.е. \( \lambda \) является корнем квадратного уравнения ниж:

\[
\nu \lambda^2 - (\nu - \omega \xi) \lambda + \xi = 0.
\]
(15)

Из (15) непосредственно следует утверждение Теоремы 1. 

Заметим, что когда \( \omega = -1 \), корнями (15) 
являются \( \lambda = 1 \) и \( \lambda = \xi/\nu \). 
П
рактический вывод, который следует из данной теоремы, 
следующий. 
Если \( \omega = -1 \) и \( \xi = \nu \), 
то все собственные числа предобусловленной матрицы 
\( B^{-1}(\omega_1, \omega_2)A \) равны 1, 

а это означает, что эффективность рассмотренного 
предобуславливателя матрица \( B_2 \) должна максимально 
улучшать приближенное дополнение Шура \( S \).
"""

#Пояснения
"""
**Дополнение Шура**:

Пусть у нас есть блочная матрица вида:
\[
A = \begin{bmatrix}
    X & Y \\
    Z & W
\end{bmatrix}
\]
где \( X \), \( Y \), \( Z \), и \( W \) — блоки матрицы. 
Тогда дополнением Шура для блока \( X \) является матрица:
\[
S = W - Z X^{-1} Y
\]

Дополнение Шура играет важную роль в различных алгоритмах и 
методах, связанных с блочными матрицами. В частности, оно 
часто используется в методах блочного предобуславливания и 
блочных итерационных методах.

Чтобы вычислить дополнение Шура для заданной матрицы 
\( B_2 \), нам нужно знать, как матрица разбита на блоки. 
Как только блоки определены, можно применить формулу, 
приведенную выше.

Давайте вычислим дополнение Шура для матрицы \( B_2 \), 
предполагая, что у нас уже есть блочное разбиение. Но для
этого мне нужно знать, как именно матрица \( B_2 \) 
разбита на блоки.
"""

def schur_complement(A):
    """
    Вычисляет дополнение Шура для блока X матрицы A.
    Предполагается, что A разбита на блоки следующим образом:
    A = | X Y |
        | Z W |
    """
    # Разделение матрицы A на блоки
    n = A.shape[0] // 2
    X = A[:n, :n]
    Y = A[:n, n:]
    Z = A[n:, :n]
    W = A[n:, n:]
    
    # Вычисление дополнения Шура
    S = W - Z @ np.linalg.inv(X) @ Y
    return S

# Пример блочной матрицы 4x4
A_example = np.array([[1, 2, 1, 0],
                      [3, 4, 0, 1],
                      [0, 1, 5, 6],
                      [7, 8, 2, 3]])

schur_complement_result = schur_complement(A_example)
print(schur_complement_result)
