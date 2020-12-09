import sys
import numpy as np

from sporfdata import consistency

from proglearn.progressive_learner import ProgressiveLearner
from proglearn.voters import TreeClassificationVoter
from proglearn.transformers import TreeClassificationTransformer
from proglearn.transformers import ObliqueTreeClassificationTransformer
from proglearn.deciders import SimpleArgmaxAverage

from sklearn.model_selection import train_test_split, cross_val_score

# Cross validates a model, returning cohen's kappa and SEM
def test(data_fn, reps, n_trees, 
        default_transformer_class, default_transformer_kwargs,
        n):

    
    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = {}

    default_decider_class = SimpleArgmaxAverage
    
    err = np.zeros(reps)
    for i in range(reps):

        X_train, X_test, y_train, y_test, n_classes = load_data(data_fn, n)
        default_decider_kwargs = {"classes" : np.arange(n_classes)}

        pl = ProgressiveLearner(
                default_transformer_class = default_transformer_class,
                default_transformer_kwargs = default_transformer_kwargs,
                default_voter_class = default_voter_class,
                default_voter_kwargs = default_voter_kwargs,
                default_decider_class = default_decider_class,
                default_decider_kwargs = default_decider_kwargs)
      
        pl.add_task(X_train, y_train, num_transformers=n_trees)

        y_hat = pl.predict(X_test, task_id=0)

        err[i] = 100 - (np.sum(y_test == y_hat) / len(y_test)) * 100
        print(err[i])

    return np.mean(err), np.std(err)

def load_data(data_fn, n):

    X, y = data_fn(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

    return X_train, X_test, y_train, y_test, len(np.unique(y))


def main():
   
    # Parameters
    n = 2000
    feature_combinations = 1
    density = 1
   
    # Fixed params
    max_depth = np.inf
    reps = 5
    n_trees = 10

    rf_kwargs = {"kwargs" : {
                             #"max_depth" : max_depth,
                             #"max_features" : int(p/feature_combinations)
                            }
                }

    of_kwargs = {"kwargs" : {"max_depth" : max_depth,
                             "feature_combinations" : feature_combinations,
                             "density" : density}}

    RF_err, RF_std = test(consistency, reps, n_trees,
                            default_transformer_class=TreeClassificationTransformer,
                            default_transformer_kwargs=rf_kwargs,
                            n=n)
    
    print("RF err, RF std:", RF_err, RF_std)

    sys.exit(0)
    OF_err, OF_std = test(consistency, reps, n_trees,
                            default_transformer_class=ObliqueTreeClassificationTransformer,
                            default_transformer_kwargs=of_kwargs,
                            n=n)
    
    print("OF err, OF std:", OF_err, OF_std)

if __name__ == "__main__":
    main()

