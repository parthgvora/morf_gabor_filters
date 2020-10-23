import numpy as np
import matplotlib.pyplot as plt

from rerf.rerfClassifier import rerfClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ObliqueForestClassifier

from sporfdata import sparseparity, orthant, trunk

def visualize_data():
    x, y = sparseparity(1000, 2, 2)
    colours = {0:"red", 1:"blue"}
    colourmap = [colours[i] for i in y]
    plt.scatter(x[:, 0], x[:, 1], color=colourmap)
    plt.savefig("sparse_parity")

    x, y = orthant(1000, 2)
    colours = {0:"red", 1:"blue", 2:"green", 3:"yellow"}
    colourmap = [colours[i] for i in y]
    plt.scatter(x[:, 0], x[:, 1], color=colourmap)
    plt.savefig("orthant")


    x, y = trunk(1000, 2)
    colours = {0:"red", 1:"blue"}
    colourmap = [colours[i] for i in y]
    plt.scatter(x[:, 0], x[:, 1], color=colourmap)
    plt.savefig("trunk")


def multitest(data, n, iters, clf, **clf_kwargs):

    acc = np.zeros(iters)
    for i in range(0, iters):

        c = clf(**clf_kwargs)
        X_train, y_train = data(n)
        X_test, y_test = data(n)

        c.fit(X_train, y_train)
        y_hat = c.predict(X_test)
        acc[i] = np.sum(y_hat == y_test) / n
    
    return np.mean(acc), np.std(acc)


def test_RF():

    kwargs = {
            "n_estimators" : 100
    }

    print("Sparse Parity")
    for n in [1000, 5000, 10000]:
        sparse_acc, sparse_std = multitest(sparseparity, n, 3, RandomForestClassifier, **kwargs)
        print(n, sparse_acc)

    print("Orthant")
    for n in [400, 2000, 4000]:
        orth_acc, orth_std = multitest(orthant, n, 3, RandomForestClassifier, **kwargs)
        print(n, orth_acc)

def test_RerF():

    kwargs = {
            "n_estimators" : 100,
            "projection_matrix": "RerF",
            "feature_combinations": 1.5
    }
    
    print("Sparse Parity")
    for n in [1000, 5000, 10000]:
        kwargs["feature_combinations"] = 3 
        sparse_acc, sparse_std = multitest(sparseparity, n, 3, rerfClassifier, **kwargs)
        print(n, sparse_acc)

    print("Orthant")
    for n in [400, 2000, 4000]:
        orth_acc, orth_std = multitest(orthant, n, 3, rerfClassifier, **kwargs)
        print(n, orth_acc)

def test_OF():

    kwargs = {
            "n_estimators" : 100,
            "feature_combinations": 1.5,
            "density": 1
    }
    
    print("Sparse Parity")
    for n in [1000, 5000, 10000]:
        kwargs["density"] = 3/20
        sparse_acc, sparse_std = multitest(sparseparity, n, 3, ObliqueForestClassifier, **kwargs)
        print(n, sparse_acc)

    print("Orthant")
    for n in [400, 2000, 4000]:
        kwargs["density"] = 3/6 
        orth_acc, orth_std = multitest(orthant, n, 3, ObliqueForestClassifier, **kwargs)
        print(orth_acc)



def main():

    #visualize_data()
    
    print("Random Forest")
    test_RF()

    print("Rerf")
    test_RerF()

    print("Oblique Forest")
    test_OF()


if __name__ == "__main__":
    main()
