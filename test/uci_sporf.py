import sys
import numpy as np
import pandas as pd

from proglearn.progressive_learner import ProgressiveLearner
from proglearn.voters import TreeClassificationVoter
from proglearn.transformers import TreeClassificationTransformer
from proglearn.transformers import ObliqueTreeClassificationTransformer
from proglearn.deciders import SimpleAverage

from sklearn.model_selection import train_test_split, cross_val_score

# Cross validates a model, returning cohen's kappa and SEM
def test(data_file, reps, n_trees, 
        default_transformer_class, default_transformer_kwargs):

    
    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = {}

    default_decider_class = SimpleAverage
    
    kappa = np.zeros(reps)
    for i in range(reps):

        X_train, X_test, y_train, y_test, n_classes = load_data(data_file)
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

        acc = np.sum(y_test == y_hat) / len(y_test)

        chance_pred = 1 / n_classes
        kappa[i] = (acc - chance_prd) / (1 - chance_pred)

    return np.mean(kappa) * 100, (np.std(kappa) * 100) / np.sqrt(reps)

def load_data(data_file):

    df = pd.read_csv(data_file)
    X = df[df.columns[:-1]].to_numpy()
    y = df[df.columns[-1]].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

    return X_train, X_test, y_train, y_test, len(np.unique(y))

def main():
   
    data_files = [
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/hill-valley/Hill_Valley_without_noise_Testing.data" 
                 ]

    # Parameters
    max_depth = 10
    feature_combinations = 2
    density = 0.2

    reps = 5
    n_trees = 10

    rf_kwargs = {"kwargs" : {"max_depth" : max_depth} }
    of_kwargs = {"kwargs" : {"max_depth" : max_depth,
                             "feature_combinations" : feature_combinations,
                             "density" : density}}

    RF_acc, RF_std = test(data_files[0], reps, n_trees,
                            default_transformer_class=TreeClassificationTransformer,
                            default_transformer_kwargs=rf_kwargs)

    OF_acc, OF_std = test(data_files[0], reps, n_trees,
                            default_transformer_class=ObliqueTreeClassificationTransformer,
                            default_transformer_kwargs=of_kwargs)

    print("RF acc, RF std:", RF_acc, RF_std)
    print("OF acc, OF std:", OF_acc, OF_std)

if __name__ == "__main__":
    main()

