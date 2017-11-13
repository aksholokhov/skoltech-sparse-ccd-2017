import timeit
from scipy import sparse
import numpy as np
from scipy.sparse.linalg import norm
import fibonacci_heap_mod as fhm



def f(x, X, y, mu):
    return sparse.linalg.norm(X.dot(x) - y)**2 + mu/2*sparse.linalg.norm(x)**2

def g(x, X, y, mu):
    return 2*X.T.dot(X.dot(x.T) - y) + mu*x.T



def CCD_sparse(X, y, mu, x0, e, k_max=1e5, history_elements = ("g_norm", "d_sparsity", "time", "f")):
    n = max(x0.shape)

    history = {}
    for element in history_elements:
        history[element] = []

    A = sparse.hstack([y, -X]).tocsr()
    I = sparse.eye(n+1).tolil()
    I[0][0] = 0
    I = I.tocsr()
    H = (2*(A.T).dot(A) + I.multiply(mu)).T
    x0_ext = sparse.hstack([sparse.eye(1), x0]).tocsr()

    g_elems = []
    heap = fhm.Fibonacci_heap()

    g_x = H.dot(x0_ext.T).T

    g_norm = 0
    if n <= 1e7:    # dense vectors work significantly better if not blow memory
        for i, val in enumerate(np.squeeze(g_x.toarray())):
            if i == 0: continue
            g_elems.append(heap.enqueue(i, val))
            g_norm += val**2
    else:
        # DOESN'T WORK
        k = 1
        t = 0
        g_x = g_x.tolil()
        for i in sorted(g_x.nonzero()[0][1:]):
            while k < i:
                g_elems.append(heap.enqueue(k, 0))
                k += 1
            g_elems.append(heap.enqueue(i, g_x[i, 0]))
            k = i+1
            t+= 1

    g_norm_init = g_norm
    beta = 1
    z = (x0 / beta).tolil()

    start = timeit.default_timer()

    for i in range(1, int(k_max)):
        min_coord = heap.min().get_value()
        if min_coord != np.argmin(np.squeeze(g_x.toarray())[1:]) + 1:
            print("still fail")

        if g_norm <= e*g_norm_init:
            return z * beta, "success", history

        gamma = 1/(i + 1)     #константный шаг
        beta *= (1 - gamma)
        gamma_n = gamma / beta

        delta_grad = (gamma * H[min_coord] - gamma*g_x + gamma*H[0]).tolil()

        if "g_norm" in history_elements:
            history["g_norm"].append(g_norm)
        if "f" in history_elements:
            x = beta * z
            history["f"].append(f(x.T, X, y, mu))
        if "time" in history_elements:
            history["time"].append(timeit.default_timer() - start)
        if "d_sparsity" in history_elements:
            history["d_sparsity"].append(len(delta_grad.nonzero()[1]))

        for k in delta_grad.nonzero()[1]:
            k -= 1
            if k < 0: continue
            if k == 100: print("azaza")
            old_priority = g_elems[k].get_priority()
            new_priority = old_priority + delta_grad[0, k+1]
            value = g_elems[k].get_value()
            heap.decrease_key(entry=g_elems[k], new_priority=heap.min().get_priority() - 1)
            heap.dequeue_min()
            g_elems[k] = heap.enqueue(value=value, priority=new_priority)
            g_norm = g_norm - old_priority**2 + new_priority**2

        z[0, min_coord-1] += gamma_n
        g_x += delta_grad

    return z * beta, "iterations_exceeded", history