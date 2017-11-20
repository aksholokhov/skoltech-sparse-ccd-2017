from scipy import sparse
import numpy as np
from lib.optimization import noname_algorithm_ridge


def f(x, X, y, mu):
    return sparse.linalg.norm(X.dot(x.T) - y, ord='fro')**2 + mu/2*sparse.linalg.norm(x, ord='fro')**2


if __name__ == "__main__":
    n = int(1e5)
    m = n
    mu = 1/n
    X = sparse.rand(m, n, density=1/n).tocsr()
    x_true = sparse.rand(1, n, density=0.001).tocsr()
    #X = arora_sparsification(array(X), 15*e_lower_lim)
    x_true /= x_true.sum()
    y = X.dot(x_true.T)
    e_lower_lim = X.max()
    print("%.6f%% of non-zero elements"%(100*len(X.nonzero()[1])/(n*m)))
    print("%d non-zero elements"%(len(X.nonzero()[1])))
    x0 = sparse.rand(1, n, density=1/n).tocsr()
    x0 /= x0.sum()

    x, message, history = noname_algorithm_ridge(X, y, mu, x0, e=1e-3, k_max=20, step="parabolic")

    #print(message)