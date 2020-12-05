import sys
import numpy as np

from sporfdata import sparse_parity

from proglearn.progressive_learner import ProgressiveLearner
from proglearn.voters import TreeClassificationVoter
from proglearn.transformers import TreeClassificationTransformer
from proglearn.transformers import ObliqueTreeClassificationTransformer
from proglearn.deciders import SimpleArgmaxAverage

from sklearn.model_selection import train_test_split, cross_val_score

# Cross validates a model, returning cohen's kappa and SEM
def test(data_fn, reps, n_trees, 
        default_transformer_class, default_transformer_kwargs,
        n, p):

    
    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = {}

    default_decider_class = SimpleArgmaxAverage
    
    #kappa = np.zeros(reps)
    err = np.zeros(reps)
    for i in range(reps):

        X_train, X_test, y_train, y_test, n_classes = load_data(data_fn, n, p)
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

        #chance_pred = 1 / n_classes
        #kappa[i] = (acc - chance_pred) / (1 - chance_pred)

        #mean_k = np.mean(kappa) * 100
        #mean_std = np.std(kappa) * 100


    return np.mean(err), np.std(err)

def load_data(data_fn, n, p):

    # want curve for 250, 500, 1000, 2000, p*=3, p=10
    X, y = data_fn(n, p);
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
    return X_train, X_test, y_train, y_test, len(np.unique(y))


def main():
   
    # Parameters
    n = 250
    p = 20
    feature_combinations = 2
    density = 0.1
   
    # Fixed params
    max_depth = np.inf
    reps = 3
    n_trees = 20 #keep this fixed. works for 1000, 5000 on RF

    rf_kwargs = {"kwargs" : {#"max_depth" : max_depth,
                             "max_features" : int(p/feature_combinations)}}

    of_kwargs = {"kwargs" : {"max_depth" : max_depth,
                             "feature_combinations" : feature_combinations,
                             "density" : density}}

    RF_err, RF_std = test(sparse_parity, reps, n_trees,
                            default_transformer_class=TreeClassificationTransformer,
                            default_transformer_kwargs=rf_kwargs,
                            n=n, p=p)
    
    print("RF err, RF std:", RF_err, RF_std)

    OF_err, OF_std = test(sparse_parity, reps, n_trees,
                            default_transformer_class=ObliqueTreeClassificationTransformer,
                            default_transformer_kwargs=of_kwargs,
                            n=n, p=p)
    
    print("OF err, OF std:", OF_err, OF_std)

if __name__ == "__main__":
    main()

