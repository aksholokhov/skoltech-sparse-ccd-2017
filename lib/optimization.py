import timeit
from scipy import sparse
import numpy as np
from scipy.sparse.linalg import norm
import fibonacci_heap_mod as fhm



def f(x, X, y, mu):
    return sparse.linalg.norm(X.dot(x.T) - y)**2 + mu/2*sparse.linalg.norm(x)**2

def g(x, X, y, mu):
    return 2*X.T.dot(X.dot(x.T) - y) + mu*x.T


def CCD_sparse(X, y, mu, x0, e, k_max = 1e5, mode = "heap", step = "constant",
               history_elements = ("g_norm", "d_sparsity", "time", "f", "gamma", "f_approx")):
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

    g_x = H.dot(x0_ext.T).T
    g_norm = 0
    popr = g_x - H[0]
    f_x = f(x0, X, y, mu)

    AT = A.T
    Ax = A.dot(x0_ext.T).T
    popr_Ax = Ax - AT[0]

    if mode is "heap":
        g_elems = []
        heap = fhm.Fibonacci_heap()

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

    else:
        g_norm = sparse.linalg.norm(g_x)**2 - g_x[0, 0]**2

    g_norm_init = g_norm
    beta = 1
    z = (x0_ext / beta).tolil()

    start = timeit.default_timer()

    for i in range(1, int(k_max)):
        if mode is "heap":
            min_coord = heap.min().get_value()
        else:
            min_coord = g_x[0, 1:].argmin() + 1

        if g_norm <= e*g_norm_init:
            return z[0, 1:] * beta, "success", history

        if step is "parabolic":
            alpha = 0.001
            x = beta*z
            #f_x = f(x[:, 1:], X, y, mu)
            f_x1 = (1 - alpha) ** 2 * f_x + 2 * alpha * (1 - alpha) * (Ax[0, 0] + Ax[0, min_coord]) + \
                    + (2 + mu) * alpha ** 2 + mu * alpha * (1 - alpha) * (1 + x[0, min_coord])
            alpha *= 2
            f_x2 = (1 - alpha) ** 2 * f_x + 2 * alpha * (1 - alpha) * (Ax[0, 0] + Ax[0, min_coord]) + \
                    + (2 + mu) * alpha ** 2 + mu * alpha * (1 - alpha) * (1 + x[0, min_coord])
            gamma = - 0.5*alpha*(4*f_x1 - 3*f_x - f_x2)/(f_x2 - 2*f_x1 + f_x)

            if abs(gamma) >= 1:
                gamma = np.sign(gamma)*0.99

            f_x = (1 - gamma) ** 2 * f_x + 2 * gamma * (1 - gamma) * (Ax[0, 0] + Ax[0, min_coord]) + \
                      + (2 + mu) * gamma ** 2 + mu * alpha * (1 - gamma) * (1 + x[0, min_coord])
            print(f_x)
            t = x.copy()*(1-gamma)
            t[0, min_coord] += gamma
            print(f(t[:, 1:], X, y, mu))

        else:
            gamma = 1/(i + 10)     #константный шаг

        beta *= (1 - gamma)
        gamma_n = gamma / beta

        delta_grad = gamma*(H[min_coord] - popr).tolil()

        if "g_norm" in history_elements:
            history["g_norm"].append(g_norm/g_norm_init)
        if "f" in history_elements:
            x = beta * z[0, 1:]
            history["f"].append(f(x, X, y, mu))
        if "f_approx" in history_elements:
            history["f_approx"].append(f_x)
        if "time" in history_elements:
            history["time"].append(timeit.default_timer() - start)
        if "d_sparsity" in history_elements:
            history["d_sparsity"].append(len(delta_grad.nonzero()[1]))
        if "gamma" in history_elements:
            history["gamma"].append(gamma)

        if mode is "heap":
            for k in delta_grad.nonzero()[1]:
                if k  == 0: continue
                old_priority = g_elems[k-1].get_priority()
                new_priority = old_priority + delta_grad[0, k]
                if old_priority > new_priority:
                    heap.decrease_key(entry=g_elems[k-1], new_priority=new_priority)
                else:
                    value = g_elems[k-1].get_value()
                    heap.decrease_key(entry=g_elems[k-1], new_priority=heap.min().get_priority() - 1)
                    heap.dequeue_min()
                    g_elems[k-1] = heap.enqueue(value=value, priority=new_priority)
                g_norm = g_norm - old_priority**2 + new_priority**2
        else:
            g_x += delta_grad
            g_norm = sparse.linalg.norm(g_x)**2 - g_x[0, 0]**2

        z[0, min_coord] += gamma_n

        popr += delta_grad

        delta_Ax =  gamma*(AT[min_coord] - popr_Ax)
        Ax += delta_Ax
        popr_Ax += delta_Ax

        #print("%.7f = %.7f" % (f_x_new, f(beta*z[:, 1:], X, y, mu)), '\n')

    return z[0, 1:] * beta, "iterations_exceeded", history