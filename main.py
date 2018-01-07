from scipy import sparse
import numpy as np
from lib.optimization import noname_algorithm_ridge
from lib.heaps import FibonacciHeap


if __name__ == "__main__":
    n = int(1e5)  # Размерность пространства. Осторожно: при порядке n = 10^8 памяти в 8 GB может не хватать
    m = int(1e5)
    mu = 1  # Коэффицент регуляризации в задаче Ridge
    X = sparse.rand(m, n,
                    density=20 / n).tocsr()  # Матрица исходных данных Х размера [m*n], плотность = доля ненулевых эл-тов
    row_sums = np.array(X.sum(axis=1))[:, 0]
    row_indices, col_indices = X.nonzero()
    X.data /= row_sums[row_indices]
    x_true = sparse.rand(1, n, density=0.3).tocsr()  # Искомое решение
    x_true /= x_true.sum()  # Решение живет на единичном симплексе
    # Вектор Y ответов в регрессии искусствdенно делается всюду плотным плотным путем зашумления
    y = X.dot(x_true.T) + sparse.rand(m, 1, density=1, format="csr") * 0.0000000001
    print("%d non-zero elements in matrix X" % (len(X.nonzero()[1])))
    print(x_true.count_nonzero(), "non-zero elements in x_true")
    x0 = sparse.rand(1, n, density=1 / n).tocsr()  # Стартовая точка -- случайный угол симплекса

    #heap = FibonacciHeap(x0)
    #m = heap.get_min()
    #print(m)

    x, message, history = noname_algorithm_ridge(X, y, mu, x0, e=1e-3, k_max=10, heap_type="fibonacci", step="constant",
                                                 history_elements=("time", "g_norm"))

    print(history["g_norm"])

    #x, message, history = noname_algorithm_ridge(X, y, mu, x0, e=1e-3, k_max=300, step="parabolic",
    #                                             history_elements=("time", "g_norm"))
    #print(message)