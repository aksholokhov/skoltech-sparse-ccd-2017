import numpy as np
from scipy import sparse


class SparsificationException(Exception):
    pass


def arora_sparsification(X, e = None, random_seed = None):
    if random_seed is not None:
        np.random.seed(random_seed)

    if e is None:
        e = X.max()

    A = sparse.lil_matrix(X.copy())

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


def bernstein_sparsification(X, e, random_seed = None, n_samples = None):
    if random_seed is not None:
        np.random.seed(random_seed)

    X_1st_norm = X.sum()
    is_positive = X.min() >= 0

    X = X.tolil()
    probabilities = [X[i, j] / X_1st_norm for (i, j) in zip(*X.nonzero())]

    if n_samples is None:
        n_samples = int(np.log(sum(X.shape)) * max(X.shape) / e ** 2)

    sparsified_indices = np.random.choice(np.arange(0, len(X.nonzero()[1])), n_samples, p=probabilities)

    indices = list(zip(*X.nonzero()))

    X_bern = sparse.lil_matrix(X.shape)

    for k in sparsified_indices:
        i, j = indices[k]
        if is_positive:
            X_bern[i, j] += 1
        else:
            X_bern[i, j] += np.sign(X[i, j])

    X_bern = X_bern.tocsr()
    X = X.tocsr()

    X_bern /= n_samples

    return X_bern

def metropolis_sparsification(X, random_seed = None):
    X = X.tolil()

    if X.shape[0] != X.shape[1]:
        raise SparsificationException("M != N (not implemented yet")

    n = X.shape[0]
    k = 0
    b_prev = None
    X_mh = sparse.lil_matrix(X.shape)
    indices = list(zip(*X.nonzero()))
    already_got = np.zeros(len(indices))
    while k <= int(n * np.log(n)):
        pt = np.random.randint(low=0, high=len(indices))
        if already_got[pt]:
            continue
        i, j = indices[pt]
        b = X[i, j]
        if b_prev == None:
            X_mh[i, j] = np.sign(b)
        else:
            if np.random.rand() < min(abs(b) / abs(b_prev), 1):
                X_mh[i, j] = np.sign(b)
            else:
                continue
        b_prev = b
        already_got[pt] = 1
        k += 1

    X = X.tocsr()
    return X_mh