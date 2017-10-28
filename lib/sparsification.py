import numpy as np
from scipy import sparse

class SparsificationException(Exception):
    pass


def arora_sparsification(M, e = None, random_seed = None):
    if random_seed is not None:
        np.random.seed(random_seed)

    if e is None:
        e = M.max()

    A = sparse.lil_matrix(M.copy())

    for i, j in zip(*A.nonzero()):
        x = A[i, j]
        if abs(x) > e:
            continue
        p = abs(x)/e
        if p > 1 or p < 0:
            raise SparsificationException("Inadequate probability on (%d, %d): %.2f"%(i, j, p))
        if np.random.rand() <= p:
            A[i, j] = np.sign(x)*e
        else:
            A[i, j] = 0

    return A