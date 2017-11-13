from scipy import sparse
from lib.optimization import CCD_sparse

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
    start_point = sparse.rand(1, n, density=100/n).tocsr()
    start_point /= start_point.sum()

    x, message, history = CCD_sparse(X, y, mu, start_point, e=1e-3, k_max=300, stoch_grad_update=True)

    print(message)