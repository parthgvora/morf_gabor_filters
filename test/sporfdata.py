import numpy as np

def sparse_parity(n, p=20, p_star=3):

    np.random.seed(12763123)

    X = np.random.uniform(-1, 1, (n, p))
    y = np.zeros(n)

    for i in range(0, n):
        y[i] = sum(X[i, :p_star] > 0) % 2;


    np.random.seed(None)
    return X, y

def orthant(n, p=6):

    orth_labels = np.asarray([2 ** i for i in range(0, p)])

    X = np.random.uniform(-1, 1, (n, p))
    y = np.zeros(n)

    for i in range(0, n):
        idx = np.where(X[i, :] < 0)[0]
        y[i] = sum(orth_labels[idx])

    return X, y

def trunk(n, p=10):

    mu_1 = np.array([1/i for i in range(1,p+1)])
    mu_0 = -1 * mu_1

    cov = np.identity(p)

    X = np.vstack((
        np.random.multivariate_normal(mu_0, cov, int(n/2)),
        np.random.multivariate_normal(mu_1, cov, int(n/2))
        ))

    y = np.concatenate((
        np.zeros(int(n/2)),
        np.ones(int(n/2))
        ))

    return X, y


