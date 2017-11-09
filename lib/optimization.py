import timeit
from scipy import sparse
import numpy as np
from scipy.sparse.linalg import norm
import fibonacci_heap_mod as fhm



def f(x, X, y, mu):
    return sparse.linalg.norm(X.dot(x) - y)**2 + mu/2*sparse.linalg.norm(x)**2


def CCD_sparse(X, y, mu, x0, e, k_max=1e5):
    n = max(x0.shape)

    A = sparse.hstack([y, -X]).tocsr()

    I = sparse.eye(n+1).tolil()
    I[0][0] = 0
    I = I.tocsr()
    H = ((A.T).dot(A) + I.multiply(mu)).T
    x0_ext = sparse.hstack([sparse.eye(1), x0]).tocsr()


    g_elems = []
    heap = fhm.Fibonacci_heap()

    if n <= 1e7:    # dense vectors work significantly better if not blow memory
        g_x = np.squeeze(H.dot(x0_ext.T).toarray())
        for i, val in enumerate(np.array(g_x)):
            if i == 0: continue
            g_elems.append(heap.enqueue(i, val))
            #if i < 100: print(i, val)
    else:
        k = 1
        t = 0
        g_x = H.dot(x0_ext.T).tolil()
        for i in sorted(g_x.nonzero()[0][1:]):
            while k < i:
                g_elems.append(heap.enqueue(k, 0))
                k += 1
            g_elems.append(heap.enqueue(i, g_x[i, 0]))
            k = i+1
            t+= 1

    beta = 1
    z = x0 / beta
    z_prev = 3 * z
    x_prev = z_prev
    fx_prev = np.inf

    start = timeit.default_timer()
    for i in range(1, int(k_max)):
        min_coord = heap.min().get_value()
        x = beta * z
        fx = f(x.T, X, y, mu)
        #x_prev = (beta / (1 - gamma)) * z_prev
        # if norm(x - x_prev) < e*norm(x):      # по относительному аргументу
        if abs(fx - fx_prev) < e:  # по относительной функции
            # if sp.sparse.linalg.norm(X_s.dot(x.T) - y_s) <= e:  # по ошибке, согласованной со sklearn
            stop = timeit.default_timer()
            return z * beta, i, stop - start

        gamma = 1/(i + 2)     #константный шаг

        beta *= (1 - gamma)
        gamma_n = gamma / beta
        z[0, min_coord] += gamma_n

        delta_grad = gamma_n * H[min_coord + 1] + gamma_n*H[0] * (1 / beta - 1 / (beta / (1 - gamma)))

        t3 = len(H[min_coord+1].nonzero()[1])
        t4 = len(H[0].nonzero()[1])

        for k, delta in zip(delta_grad.indices, delta_grad.data):
            if delta != 0 and k != 0:
                k -= 1
                new_priority = g_elems[k].get_priority() + delta
                value = g_elems[k].get_value()
                heap.decrease_key(entry=g_elems[k], new_priority=heap.min().get_priority() - 1)
                heap.dequeue_min()
                g_elems[k] = heap.enqueue(value=value, priority=new_priority)

        x_prev = x
        fx_prev = fx


    stop = timeit.default_timer()
    return z * beta, k_max, stop - start