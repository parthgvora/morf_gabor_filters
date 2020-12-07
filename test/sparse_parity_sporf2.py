import sys
import numpy as np
import pandas as pd

from proglearn.progressive_learner import ProgressiveLearner
from proglearn.voters import TreeClassificationVoter
from proglearn.transformers import ObliqueTreeClassificationTransformer
from proglearn.deciders import SimpleArgmaxAverage

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Cross validates a model, returning cohen's kappa and SEM

def testRF(n, reps, params):

    err = []
    for i in range(reps):
        X_train, X_test, y_train, y_test, n_classes = load_data(n)
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        
        err_i = 1 - (np.sum(y_hat == y_test) / len(y_hat))
        print(err_i)

        err.append(err_i)

    return np.mean(err), np.std(err)

def testOF(n, reps, n_trees, of_kwargs):

    err = np.zeros(reps)
    for i in range(reps):
        X_train, X_test, y_train, y_test, n_classes = load_data(n)

        pl = ProgressiveLearner(
                default_transformer_class = ObliqueTreeClassificationTransformer,
                default_transformer_kwargs = of_kwargs,
                default_voter_class = TreeClassificationVoter,
                default_voter_kwargs = {},
                default_decider_class = SimpleArgmaxAverage,
                default_decider_kwargs = {"classes" : np.arange(n_classes)})
      
        pl.add_task(X_train, y_train, num_transformers=n_trees)

        y_hat = pl.predict(X_test, task_id=0)

        err[i] = 1 - (np.sum(y_test == y_hat) / len(y_hat))
        print(err[i])

    return np.mean(err), np.std(err)

def load_data(n):

    # file names are flipped???
    df_train = pd.read_csv("Sparse_parity_test.csv").to_numpy()
    df_test = pd.read_csv("Sparse_parity_train.csv").to_numpy()

    idx = np.random.choice(9999, n, replace=False) 

    X_train = df_train[idx, :-1]
    y_train = df_train[idx, -1]
    X_test = df_test[:, :-1]
    y_test = df_test[:, -1]

    return X_train, X_test, y_train, y_test, 2


def main():
   
    # Parameters
    n = 1000
    feature_combinations = 3
    density = 0.1
   
    # Fixed params
    max_depth = np.inf
    reps = 3
    n_trees = 10 #keep this fixed. works for 1000, 5000 on RF

    rfparams = {
                    "n_estimators" : n_trees,
               }

    of_kwargs = { "kwargs" : {
                                "feature_combinations" : feature_combinations,
                                "density" : density
                             }
                }
    
    RF_err, RF_std = testRF(n, reps, rfparams)
    print("RF err, RF std:", RF_err, RF_std)

    OF_err, OF_std = testOF(n, reps, n_trees, of_kwargs)
    print("OF err, OF std:", OF_err, OF_std)

if __name__ == "__main__":
    main()

