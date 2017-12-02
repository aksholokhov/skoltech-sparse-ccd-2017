import timeit
from scipy import sparse
from scipy.sparse.linalg import norm
from lib.gradient_updating import RidgeHeapGradientUpdater, BasicGradientUpdater, LassoHeapGradientUpdater, GSGradientUpdater
from lib.step_size_calculation import RidgeParabolicStepSize, ConstantStepSize, CoordParabolicStepSize


def f(x, X, y, mu):
    return sparse.linalg.norm(X.dot(x.T) - y)**2 + mu/2*sparse.linalg.norm(x)**2

def g(x, X, y, mu):
    return 2*X.T.dot(X.dot(x.T) - y) + mu*x.T

def noname_algorithm_ridge(X, y, mu, x0, e, k_max = 1e5, gradient_update_mode ="heap", step ="constant",
                           history_elements = ("g_norm", "d_sparsity", "time", "f", "gamma", "f_approx", "preproc_time")):

    start = timeit.default_timer()

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
        step_size_calculator = RidgeParabolicStepSize(A, x0_ext, f_x, mu)
    elif step is "constant":
        step_size_calculator = ConstantStepSize(start_num=10)
    else:
        raise Exception("No such step size calculation mode as as %s" % step)

    if gradient_update_mode is "heap":
        grad_updater = RidgeHeapGradientUpdater(g_x[0, 1:])
    elif gradient_update_mode is "simple":
        grad_updater = BasicGradientUpdater(g_x[0, 1:])
    else:
        raise Exception("No such gradient update mode as %s" % gradient_update_mode)

    g_norm_init = grad_updater.get_norm()
    beta = 1
    z = (x0_ext / beta).tolil()

    current = timeit.default_timer()

    if "preproc_time" in history_elements:
        history["preproc_time"] = current - start

    start = timeit.default_timer()


    for i in range(1, int(k_max)):
        min_coord = grad_updater.get_coordinate() + 1

        if grad_updater.get_norm() <= e*g_norm_init:
            return z[0, 1:] * beta, "success", history

        x = beta*z
        x[0, 0] = 1
        gamma = step_size_calculator.get_step_size(x, min_coord)
        #gamma_true = step_size_checker.get_step_size(x[0, 1:].T, min_coord)
        #print(gamma - gamma_true)

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


def noname_algorithm_lasso(X, y, mu, x0, e, k_max = 1e5, gradient_update_mode ="heap", step ="constant",
                           history_elements = ("g_norm", "d_sparsity", "time", "f", "gamma", "f_approx")):

    n = max(x0.shape)

    history = {}
    for element in history_elements:
        history[element] = []

    A = sparse.hstack([y, -X]).tocsr()
    H = 2*(A.T).dot(A).T
    x0_ext = sparse.hstack([sparse.eye(1), x0]).tocsr()

    g_x = H.dot(x0_ext.T).T
    popr = g_x - H[0]
    f_x = f(x0, X, y, mu)

    if step is "parabolic":
        raise Exception("Hasn't been implemented yet: %s"%step)
        #step_size_calculator = RidgeParabolicStepSize(A, x0_ext, f_x, mu)
    elif step is "constant":
        step_size_calculator = ConstantStepSize(start_num=10)
    else:
        raise Exception("No such step size calculation mode as as %s" % step)

    if gradient_update_mode is "heap":
        grad_updater = LassoHeapGradientUpdater(g_x[0, 1:])
    elif gradient_update_mode is "simple":
        raise Exception("Hasn't been implemented yet: %s"%gradient_update_mode)
    else:
        raise Exception("No such gradient update mode as %s" % gradient_update_mode)

    g_norm_init = grad_updater.get_norm()
    beta = 1
    z = (x0_ext / beta).tolil()

    start = timeit.default_timer()

    for i in range(1, int(k_max)):
        min_coord = grad_updater.get_coordinate() + 1

        if grad_updater.get_norm() <= e*g_norm_init:
            return z[0, 1:] * beta, "success", history

        x = beta*z
        x[0, 0] = 1

        gamma = step_size_calculator.get_step_size(x, min_coord%n)
        beta *= (1 - gamma)
        gamma_n = gamma / beta
        if min_coord >= 0:
            delta_grad = gamma*(H[min_coord] - popr).tolil()
            z[0, min_coord] += gamma_n
        else:
            min_coord = -1*min_coord -1
            delta_grad = gamma*(-H[min_coord] - popr).tolil()
            z[0, min_coord] += gamma_n

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

        popr += delta_grad

    return z[0, 1:] * beta, "iterations_exceeded", history
