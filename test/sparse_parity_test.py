import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rerf.rerfClassifier import rerfClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from proglearn.forest import LifelongClassificationForest

def test(reps, samples):

    # Params
    RF_params = {
            "n_estimators" : 100,
            "projection_matrix": "Base",
            "n_jobs": -1
    }

    # These parameters are good
    RerF_params = {
            "n_estimators" : 100,
            "projection_matrix": "RerF",
            "feature_combinations": 3,
            "max_features" : 8,
            "n_jobs": -1
    }

    # These parameters also appear good
    OF_params = {
            "oblique": True,
            "default_feature_combinations": 1.25, #10/8 = 1.25
            "default_density": 0.4, # 3/8 ~= .4
            "default_n_estimators": 100,
    }

    # Samples 
    RF_trials = []
    RerF_trials = []
    OF_trials = []

    RF_stdev = []
    RerF_stdev = []
    OF_stdev = []

    for n in samples:
        print(n)
        X_train, y_train, X_test, y_test = load_data(n)
        RF_err = []
        RerF_err = []
        OF_err = []


        for r in range(reps):
            RF = rerfClassifier(**RF_params)
            RF.fit(X_train, y_train)
            y_hat = RF.predict(X_test)
            RF_err.append(1 - np.sum(y_hat == y_test) / len(y_test))

            RerF = rerfClassifier(**RerF_params)
            RerF.fit(X_train, y_train)
            y_hat = RerF.predict(X_test)
            RerF_err.append(1 - np.sum(y_hat == y_test) / len(y_test))

            OF = LifelongClassificationForest(**OF_params)
            OF.add_task(X_train, y_train)
            y_hat = OF.predict(X_test, task_id=0)
            OF_err.append(1 - np.sum(y_hat == y_test) / len(y_test))
            

        RF_trials.append(np.mean(RF_err))
        RerF_trials.append(np.mean(RerF_err))
        OF_trials.append(np.mean(OF_err))

        RF_stdev.append(np.std(RF_err))
        RerF_stdev.append(np.std(RerF_err))
        OF_stdev.append(np.std(OF_err))

    return RF_trials, RF_stdev, RerF_trials, RerF_stdev, OF_trials, OF_stdev
    

    """
    # Grid search cause fuck htis
    grid = np.zeros((11, 11))
    X_train, y_train, X_test, y_test = load_data(1000)
    for i in range(1, 11):
        for j in range(1, 11):
            RerF_params = {
                "n_estimators": 100,
                "projection_matrix": "RerF",
                "feature_combinations": i,
                "max_features": j,
                "n_jobs": -1
            }

            err = 0
            for r in range(3):
                RerF = rerfClassifier(**RerF_params)
                RerF.fit(X_train, y_train)
                y_hat = RerF.predict(X_test)
                err += 1 - np.sum(y_hat == y_test) / len(y_test)

            err /= 3
            grid[i, j] = err

    print(grid)
    """

def load_data(n):

    # Files were labelled backwards
    df = pd.read_csv("Sparse_parity_test.csv", header=None)
    X_train, y_train = df[df.columns[:-1]].to_numpy(), df[df.columns[-1]].to_numpy()

    df = pd.read_csv("Sparse_parity_train.csv", header=None)
    X_test, y_test = df[df.columns[:-1]].to_numpy(), df[df.columns[-1]].to_numpy() 

    return X_train[:n], y_train[:n], X_test, y_test

def main():
    
    samples = [500, 1000, 2000]
    reps = 3

    RF_trials, RF_stdev, RerF_trials, RerF_stdev, OF_trials, OF_stdev = test(reps, samples)

    plt.figure()
    plt.errorbar(samples, RF_trials, RF_stdev, label="Random Forest")
    plt.errorbar(samples, RerF_trials, RerF_stdev, label="RerF SPORF")
    plt.errorbar(samples, OF_trials, OF_stdev, label="Proglearn SPORF")
    plt.legend()
    plt.savefig("sparse_parity_results")

if __name__ == "__main__":
    main()
