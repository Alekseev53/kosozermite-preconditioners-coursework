
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

def B_omega_parametric(omega1, omega2, B_e, K_l, K_u):
    """
    Вычисление матрицы B(ω1, ω2) по заданной формуле.
    """
    return np.linalg.inv(B_e + omega1 * K_l) @ (B_e + omega2 * K_u)

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

# Простой пример использования
# Начальные параметры
n = 4
v0 = np.array([1, 1, 1, 1])
h = 0.1
tau = 0.01
B_e_example = np.eye(n)  # Просто единичная матрица в качестве примера

# Применение метода
v_sequence = PTTKM_method(v0, h, tau, A_example, \
                          np.array([1, 2, 3, 4]), B_e_example, \
                          K_l_example, K_u_example)

# Возвращаем первые 5 элементов последовательности для демонстрации
print(v_sequence[:5])

#Третий лист

#Четвертый лист

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

schur_complement_result = schur_complement(A_example)
print(schur_complement_result)
