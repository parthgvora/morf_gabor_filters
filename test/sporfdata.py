import numpy as np


def find_label(x):

    base = 0.0
    add = 0.5
    top = base + add

    # Locate x's position
    while x > top:
        base = top
        add /= 2
        top += add

    # at this point, base < x < top
    if x < (base + top) / 2 :
        return 0

    return 1

def consistency(n):

    X = np.zeros((n, 2))
    y = np.zeros(n)

    for i in range(0, n):

        box = np.random.randint(3)

        if box == 0:

            X[i, 0] = np.random.uniform(0, 1)
            X[i, 1] = np.random.uniform(0, 1)

            y[i] = find_label(X[i, 0])

        elif box == 1:

            X[i, 0] = np.random.uniform(1, 2)
            X[i, 1] = np.random.uniform(1, 2)
            
            if (X[i, 0] < 1.5 and X[i, 1] > 1.5) or (X[i, 0] > 1.5 and X[i, 1] < 1.5):
                y[i] = 1
            else:
                y[i] = 0

        else:

            X[i, 0] = np.random.uniform(0, 1)
            X[i, 1] = np.random.uniform(0, 1)
            
            y[i] = find_label(1 - X[i, 1])

            X[i] += 2
        
    return X, y



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

