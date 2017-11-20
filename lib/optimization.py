import timeit
from scipy import sparse
import numpy as np
from scipy.sparse.linalg import norm
import fibonacci_heap_mod as fhm
from lib.gradient_updating import HeapGradientUpdater, BasicGradientUpdater



def f(x, X, y, mu):
    return sparse.linalg.norm(X.dot(x.T) - y)**2 + mu/2*sparse.linalg.norm(x)**2

def g(x, X, y, mu):
    return 2*X.T.dot(X.dot(x.T) - y) + mu*x.T


def noname_algorithm_ridge(X, y, mu, x0, e, k_max = 1e5, gradient_update_mode ="heap", step ="constant",
                           history_elements = ("g_norm", "d_sparsity", "time", "f", "gamma", "f_approx")):

    def f_move(alpha, xAh, A, mu, x, j, yTy, fx):
        result = 2 * alpha * (1 - alpha) * xAh
        result += alpha ** 2 * (yTy + 2 * A[j].dot(A[0].T) + A[j].dot(A[j].T))
        return result[0, 0] - mu*alpha + mu/2*alpha**2 + mu/2*(alpha**2*2) + mu/2*(2 * alpha * (1 - alpha) * (1 + x[0, j])) + (1-alpha)**2*fx

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
    popr = g_x - H[0]
    f_x = f(x0, X, y, mu)

    if step is "parabolic":
        alpha = 0.1
        AT = A.T
        Ax = A.dot(x0_ext.T).T
        yTy = AT[0].dot(AT[0].T)
        prev_min_coord = None

    if gradient_update_mode is "heap":
        grad_updater = HeapGradientUpdater(g_x[0, 1:])
    elif gradient_update_mode is "simple":
        grad_updater = BasicGradientUpdater(g_x[0, 1:])
    else:
        raise Exception("No such gradient update mode as %s" % (gradient_update_mode))


    g_norm_init = grad_updater.get_norm()
    beta = 1
    z = (x0_ext / beta).tolil()

    start = timeit.default_timer()

    for i in range(1, int(k_max)):
        min_coord = grad_updater.get_coordinate() + 1

        if grad_updater.get_norm() <= e*g_norm_init:
            return z[0, 1:] * beta, "success", history

        if step is "parabolic":
            x = beta*z
            x[0, 0] = 1

            # this code in comments is still here for testing purpose only

            #h = sparse.csr_matrix((1, n+1))
            #h[0, 0] = 1
            #h[0, min_coord] = 1

            if prev_min_coord is not None:
                Ax = (1 - gamma) * Ax + gamma * (AT[0] + AT[prev_min_coord])

            Ah = AT[min_coord] + AT[0]
            xAh = Ax.dot(Ah.T)

            f_x1 = f_move(alpha, xAh, AT, mu, x, min_coord, yTy, f_x)
            f_x2 = f_move(2*alpha, xAh, AT, mu, x, min_coord, yTy, f_x)

            #f_x = f(x[0, 1:], X, y, mu)
            #f_x1_true = f(((1 - alpha)*x + alpha*h)[0, 1:], X, y, mu)
            #f_x2_true = f(((1 - 2*alpha) * x + 2*alpha * h)[0, 1:], X, y, mu)
            #f_x1 = f_x1_true
            #f_x2 = f_x2_true

            #delta = f_x1 - f_x1_true
            #delta_2 = f_x2 - f_x2_true
            #print(delta, delta_2)

            gamma = - 0.5*alpha*(4*f_x1 - 3*f_x - f_x2)/(f_x2 - 2*f_x1 + f_x)
            f_x = f_move(gamma, xAh, AT, mu, x, min_coord, yTy, f_x)

            if abs(gamma) >= 1:
                gamma = np.sign(gamma)*0.99

            prev_min_coord = min_coord

        else:
            gamma = 1/(i + 2)     #константный шаг

        beta *= (1 - gamma)
        gamma_n = gamma / beta

        delta_grad = gamma*(H[min_coord] - popr).tolil()

        grad_updater.update(delta_grad[0, 1:])

        if "g_norm" in history_elements:
            history["g_norm"].append(grad_updater.get_norm()/g_norm_init)
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
        if "x_sparsity" in history_elements:
            history["x_sparsity"].append(z.count_nonzero())
        if "x_norm" in history_elements:
            history["x_norm"].append(sparse.linalg.norm(beta*z[0, 1:]))

        z[0, min_coord] += gamma_n
        popr += delta_grad


    return z[0, 1:] * beta, "iterations_exceeded", history