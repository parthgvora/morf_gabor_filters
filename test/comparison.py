import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rerf.rerfClassifier import rerfClassifier

from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import ObliqueForestClassifier
from sklearn.model_selection import train_test_split

from sporfdata import *
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.voters import TreeClassificationVoter
from proglearn.transformers import TreeClassificationTransformer
from proglearn.transformers import ObliqueTreeClassificationTransformer
from proglearn.deciders import SimpleArgmaxAverage

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

def visualize_consistency():
    """
    Consistency
    """
    x, y = consistency(10000)
    colours = {0:"red", 1:"blue"}
    colourmap = [colours[i] for i in y]
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], color=colourmap, s=1)
    plt.savefig("consistency")

def get_data(n):
    
    X_train, y_train = consistency(n)
    X_test, y_test = consistency(1000)
    
    """
    df_train = pd.read_csv("Sparse_parity_test.csv").to_numpy()
    df_test = pd.read_csv("Sparse_parity_train.csv").to_numpy()

    idx = np.random.choice(9999, n, replace=False)

    X_train = df_train[idx, :-1]
    y_train = df_train[idx, -1]
    X_test = df_test[:, :-1]
    y_test = df_test[:, -1]
    """
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


def test_RF(N):

    kwargs = {
            "n_estimators" : 10
    }

    acc, std = [], []
    print("Sparse Parity")
    for n in N:
        sparse_acc, sparse_std = multitest(n, 3, RandomForestClassifier, **kwargs)
        acc.append(sparse_acc)
        std.append(sparse_std)

    return acc, std


def test_RerF(N):

    kwargs = {
            "n_estimators" : 10,
            "projection_matrix": "RerF",
            "feature_combinations": 1.5,
    }
   
    acc, std = [], []
    print("Sparse Parity")
    for n in N:
        sparse_acc, sparse_std = multitest(n, 3, rerfClassifier, **kwargs)
        acc.append(sparse_acc)
        std.append(sparse_std)

    return acc, std


def test_PLSPORF(N):
    
    acc, std = [], []

    of_kwargs = { "kwargs" : {
                                "feature_combinations" : 1.5,
                                "density" : 0.1
                             }
                }


    for n in N:

        accn = []
        for i in range(3):

            print(i, n)
            X_train, X_test, y_train, y_test = get_data(n)

            pl = ProgressiveLearner(
                default_transformer_class = ObliqueTreeClassificationTransformer,
                default_transformer_kwargs = of_kwargs,
                default_voter_class = TreeClassificationVoter,
                default_voter_kwargs = {},
                default_decider_class = SimpleArgmaxAverage,
                default_decider_kwargs = {"classes" : np.arange(2)})

            pl.add_task(X_train, y_train, num_transformers=10)

            y_hat = pl.predict(X_test, task_id=0)
            accn.append(1 - np.sum(y_hat == y_test)/len(y_test))
        
        acc.append(np.mean(accn))
        std.append(np.std(accn))

    return acc, std

def RFvsRerF():

    N = [1000, 5000, 9999]
    
    print("Random Forest")
    RF_acc, RF_std = test_RF(N)

    print("Rerf")
    Rerf_acc, Rerf_std = test_RerF(N)


    plt.figure()
    plt.plot([1, 3, 4], RF_acc, label="RF", color="Blue")
    plt.errorbar([1, 3, 4], RF_acc, yerr=RF_std, ls="None", color="Blue")
    plt.plot([1, 3, 4], Rerf_acc, label="RerF", color="Green")
    plt.errorbar([1, 3, 4], Rerf_acc, yerr=Rerf_std, ls="None", color="Green")
    plt.xticks([1, 3, 4], [1000, 5000, 10000])
    plt.xlabel("Training samples")
    plt.ylabel("Error")
    plt.title("Consistency: RF vs RerF")
    plt.legend()
    plt.savefig("Consistency_RFvsRerF")

def all_RFs():

    N = [200, 400, 800, 1600]
    
    print("Random Forest")
    RF_acc, RF_std = test_RF(N)

    print("Rerf")
    Rerf_acc, Rerf_std = test_RerF(N)

    print("PL SPORF")
    plsporf_acc, plsporf_std = test_PLSPORF(N)
    

    xaxis = [1, 2, 3, 4]

    plt.figure()
    
    plt.plot(xaxis, RF_acc, label="RF", color="Blue")
    plt.errorbar(xaxis, RF_acc, yerr=RF_std, ls="None", color="Blue")
    plt.plot(xaxis, Rerf_acc, label="RerF", color="Green")
    plt.errorbar(xaxis, Rerf_acc, yerr=Rerf_std, ls="None", color="Green")
    plt.plot(xaxis, plsporf_acc, label="Proglearn SPORF", color="Red")
    plt.errorbar(xaxis, plsporf_acc, yerr=plsporf_std, ls="None", color="Red")
     
    plt.xticks(xaxis, N)
    plt.xlabel("Training samples")
    plt.ylabel("Error")
    plt.title("Sparse parity: RF vs RerF vs Proglearn SPORF")
    plt.legend()
    plt.savefig("sparse_parity_all")

    
def main():
    
    #visualize_consistency()
    RFvsRerF()
    #all_RFs()

if __name__ == "__main__":
    main()
