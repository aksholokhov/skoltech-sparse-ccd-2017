import timeit
from scipy import sparse
import numpy as np
import fibonacci_heap_mod as fhm

def CCD_sparse(X, y, mu, x0, e, k_max=1e5):
    def f(x, X, y, mu):
        return sparse.linalg.norm(X.dot(x) - y) ** 2 + mu / 2 * sparse.linalg.norm(x) ** 2

    def grad_f(x, X, XT, y, mu):
        return 2 * XT.dot(X.dot(x) - y) + mu * x

    n = len(x0)

    steps = [x0]

    gammas = []

    X_s = sparse.csr_matrix(X)
    X_t_s = sparse.csr_matrix(X.T)
    y_s = sparse.csr_matrix(y).T

    A = sparse.csr_matrix(np.c_[y, -X])
    t = np.eye(n+1)
    t[0][0] = 0
    I = sparse.csr_matrix(t)
    H = ((A.T).dot(A) + I.multiply(mu)).T

    x1 = np.ones(n + 1)
    x1[1:] = x0
    x1 = sparse.csr_matrix(x1)
    g_x = H.dot(x1.T).todense()[1:]

    heap = fhm.Fibonacci_heap()
    g_elems = []

    for i, val in enumerate(np.array(g_x)):
        g_elems.append(heap.enqueue(i, val))

    x = x0
    beta = 1
    z = x0 / beta
    z_prev = 3 * z
    gamma = 0.5
    g0 = np.linalg.norm(grad_f(x0, X_s, X_t_s, y, mu))
    start = timeit.default_timer()
    for i in range(1, int(k_max)):
        min_coord = heap.min().get_value()
        d = np.zeros(n)
        d[min_coord] = 1
        d = sparse.csr_matrix(d)

        x = sparse.csr_matrix(beta * z)
        x_prev = sparse.csr_matrix((beta / (1 - gamma)) * z_prev)
        # if norm(x - x_prev) < e*norm(x):      # по относительному аргументу
        if e >= abs(f(x.T, X_s, y_s, mu) - f(x_prev.T, X_s, y_s, mu)):  # по относительной функции

            # if sp.sparse.linalg.norm(X_s.dot(x.T) - y_s) <= e:  # по ошибке, согласованной со sklearn
            stop = timeit.default_timer()
            return z * beta, i, stop - start, gammas

        # line search
        gamma_max = 1
        armiho_eps = 0.9
        armiho_theta = 0.9
        gamma *= 2
        while f(((1 - gamma) * x + gamma * d).T, X_s, y_s, mu) - f(x.T, X_s, y_s, mu) > armiho_eps * gamma * grad_f(x.T,
                                                                                                                    X_s,
                                                                                                                    X_t_s,
                                                                                                                    y_s,
                                                                                                                    mu).T.dot(
                d.T):
            gamma *= armiho_theta
            if gamma < gamma_max / 10000:
                break
        # print(gamma)
        gammas.append(gamma)
        # gamma = 1/(i + 2)     #константный шаг
        beta *= (1 - gamma)
        gamma_n = gamma / beta
        z_prev = np.copy(z)
        z[min_coord] += gamma_n

        delta_grad = gamma_n * H[min_coord + 1] + H[0] * (1 / beta - 1 / (beta / (1 - gamma)))

        for k, delta in zip(delta_grad.indices, delta_grad.data):
            if delta != 0 and k != 0:
                k -= 1
                new_priority = g_elems[k].get_priority() + delta
                value = g_elems[k].get_value()
                heap.decrease_key(entry=g_elems[k], new_priority=heap.min().get_priority() - 1)
                heap.dequeue_min()
                g_elems[k] = heap.enqueue(value=value, priority=new_priority)

    stop = timeit.default_timer()
    return z * beta, k_max, stop - start, gammas