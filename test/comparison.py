import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rerf.rerfClassifier import rerfClassifier

from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import ObliqueForestClassifier
from sklearn.model_selection import train_test_split

from sporfdata import *

def visualize_data():
    """
    Sparse Parity
    """
    x, y = sparse_parity(1000)
    colours = {0:"red", 1:"blue"}
    
    x_pos, y_pos = x[x[:, 2] > 0], y[x[:, 2] > 0]
    x_neg, y_neg = x[x[:, 2] < 0], y[x[:, 2] < 0]

    plt.figure()
    plt.scatter(x_pos[:, 0], x_pos[:, 1], color=[colours[i] for i in y_pos])
    plt.savefig("sparse_parity_x3_pos")

    plt.figure()
    plt.scatter(x_neg[:, 0], x_neg[:, 1], color=[colours[i] for i in y_neg])
    plt.savefig("sparse_parity_x3_neg")
    
    """
    Orthant
    """
    x, y = orthant(1000, 3)
    colours = {0:"red", 1:"blue", 2:"green", 3:"yellow",
               4: "pink", 5:"gray", 6:"purple", 7:"brown"}

    x_pos, y_pos = x[x[:, 2] > 0], y[x[:, 2] > 0]
    x_neg, y_neg = x[x[:, 2] < 0], y[x[:, 2] < 0]

    plt.figure()
    plt.scatter(x_pos[:, 0], x_pos[:, 1], color=[colours[i] for i in y_pos])
    plt.savefig("orthant_x3_pos")

    plt.figure()
    plt.scatter(x_neg[:, 0], x_neg[:, 1], color=[colours[i] for i in y_neg])
    plt.savefig("orthant_x3_neg")
 
    """
    Trunk
    """
    x, y = trunk(1000, 2)
    colours = {0:"red", 1:"blue"}
    colourmap = [colours[i] for i in y]
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], color=colourmap)
    plt.savefig("trunk")


    """
    Consistency
    """
    x, y = consistency(1000)
    colours = {0:"red", 1:"blue"}
    colourmap = [colours[i] for i in y]
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], color=colourmap)
    plt.savefig("consistency")

def get_data(n):
    df_train = pd.read_csv("Sparse_parity_test.csv").to_numpy()
    df_test = pd.read_csv("Sparse_parity_train.csv").to_numpy()

    idx = np.random.choice(9999, n, replace=False)

    X_train = df_train[idx, :-1]
    y_train = df_train[idx, -1]
    X_test = df_test[:, :-1]
    y_test = df_test[:, -1]

    return X_train, X_test, y_train, y_test


def multitest(n, iters, clf, **clf_kwargs):

    acc = np.zeros(iters)
    for i in range(0, iters):

        c = clf(**clf_kwargs)
        X_train, X_test, y_train, y_test = get_data(n)
        
        c.fit(X_train, y_train)
        y_hat = c.predict(X_test)
        acc[i] = 1 - np.sum(y_hat == y_test) / len(y_test)
    
    return np.mean(acc), np.std(acc)


def test_RF():

    kwargs = {
            "n_estimators" : 100
    }

    acc, std = [], []
    print("Sparse Parity")
    for n in [1000, 5000, 9999]:
        sparse_acc, sparse_std = multitest(n, 3, RandomForestClassifier, **kwargs)
        acc.append(sparse_acc)
        std.append(sparse_std)

    return acc, std


def test_RerF():

    kwargs = {
            "n_estimators" : 100,
            "projection_matrix": "RerF",
            "feature_combinations": 1.5,
    }
   
    acc, std = [], []
    print("Sparse Parity")
    for n in [1000, 5000, 9999]:
        sparse_acc, sparse_std = multitest(n, 3, rerfClassifier, **kwargs)
        acc.append(sparse_acc)
        std.append(sparse_std)

    return acc, std

def main():

    #visualize_data()
    
    print("Random Forest")
    RF_acc, RF_std = test_RF()

    print("Rerf")
    Rerf_acc, Rerf_std = test_RerF()


    plt.figure()
    plt.plot([1, 3, 4], RF_acc, label="RF", color="Blue")
    plt.errorbar([1, 3, 4], RF_acc, yerr=RF_std, ls="None", color="Blue")
    plt.plot([1, 3, 4], Rerf_acc, label="RerF", color="Green")
    plt.errorbar([1, 3, 4], Rerf_acc, yerr=Rerf_std, ls="None", color="Green")
    plt.xticks([1, 3, 4], [1000, 5000, 10000])
    plt.xlabel("Training samples")
    plt.ylabel("Error")
    plt.title("Sparse parity: RF vs RerF")
    plt.legend()
    plt.savefig("Sparse_Parity_RFvsRerF2")
    #print("Oblique Forest")
    #test_OF()


if __name__ == "__main__":
    main()
