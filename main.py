from scipy import sparse
import numpy as np
from lib.optimization import CCD_sparse


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

    A = sparse.hstack([y, -X]).tocsr()
    I = sparse.eye(n + 1).tolil()
    I[0][0] = 0
    I = I.tocsr()
    H = (2 * (A.T).dot(A) + I.multiply(mu)).T
    x0_ext = sparse.hstack([sparse.eye(1), x0]).tocsr()

    g_x = H.dot(x0_ext.T).T
    g_norm = 0
    popr = g_x - H[0]
    f_x = f(x0, X, y, mu)

    min_coord = 5
    AT = A.T
    Ax = A.dot(x0_ext.T).T
    h = sparse.csr_matrix((1, n+1))
    h[0, 0] = 1
    h[0, min_coord] = 1
    Ah = A.dot(h.T).T
    popr_Ax = Ax - AT[0]

    a1 = []
    b1 = []
    c1 = []

    def f_move(alpha, xAx, xAh, A, mu, x, j, yTy):
        result = (1 - alpha) ** 2 * xAx + alpha ** 2 * (yTy + 2*A[j].dot(A[0].T) + A[j].dot(A[j].T))
        result += 2 * alpha * (1 - alpha) * xAh
        result += mu/2*((1 - alpha) ** 2 * x.dot(x.T))
        return result[0, 0] - mu/2 + mu/2*(alpha**2*2) + mu/2*(2 * alpha * (1 - alpha) * (1 + x[0, j]))

    for alpha in np.linspace(0, 1, 10):
        f_x3 = f_move(alpha, Ax.dot(Ax.T), Ax.dot(Ah.T), AT, mu, x0_ext, min_coord, AT[0].dot(AT[0].T))

        xt = x0_ext*(1-alpha) + h*alpha
        f_x2 = f(xt[0, 1:], X, y, mu)
        #a1.append(f_x1)
        b1.append(f_x2)
        c1.append(f_x3)

    #print(a1)
    #print(b1)
    #print(c1)

    x, message, history = CCD_sparse(X, y, mu, x0, e=1e-3, k_max=5, step="parabolic")

    #print(message)