import sys
import numpy as np

from proglearn.progressive_learner import ProgressiveLearner
from proglearn.voters import TreeClassificationVoter
from proglearn.transformers import TreeClassificationTransformer
from proglearn.transformers import ObliqueTreeClassificationTransformer
from proglearn.deciders import SimpleAverage

from sporfdata import sparseparity, orthant

def test_RF(DataGen, n_samples_train, n_samples_test, reps, n_trees, classes,
            max_depth):

    default_transformer_class = TreeClassificationTransformer
    default_transformer_kwargs = {"kwargs" : {"max_depth" : max_depth}}

    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = {}

    default_decider_class = SimpleAverage
    default_decider_kwargs = {"classes" : np.arange(classes)} # Change?

    acc = np.zeros(reps)
    for i in range(reps):

        X_train, y_train = DataGen(n_samples_train)
        X_test, y_test = DataGen(n_samples_test)

        pl = ProgressiveLearner(
                default_transformer_class = default_transformer_class,
                default_transformer_kwargs = default_transformer_kwargs,
                default_voter_class = default_voter_class,
                default_voter_kwargs = default_voter_kwargs,
                default_decider_class = default_decider_class,
                default_decider_kwargs = default_decider_kwargs)
      
        pl.add_task(X_train, y_train, num_transformers=n_trees)

        y_hat = pl.predict(X_test, task_id=0)

        acc[i] = np.sum(y_test == y_hat) / n_samples_test

    return np.mean(acc)

def test_SPORF(DataGen, n_samples_train, n_samples_test, reps, n_trees, classes,
               max_depth, feature_combinations, density):

    default_transformer_class = ObliqueTreeClassificationTransformer
    default_transformer_kwargs = {"kwargs" : {"max_depth" : max_depth,
                                              "feature_combinations" : feature_combinations,
                                              "density" : density }}

    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = {}

    default_decider_class = SimpleAverage
    default_decider_kwargs = {"classes" : np.arange(classes)} # Change?

    acc = np.zeros(reps)
    for i in range(reps):

        X_train, y_train = DataGen(n_samples_train)
        X_test, y_test = DataGen(n_samples_test)

        pl = ProgressiveLearner(
                default_transformer_class = default_transformer_class,
                default_transformer_kwargs = default_transformer_kwargs,
                default_voter_class = default_voter_class,
                default_voter_kwargs = default_voter_kwargs,
                default_decider_class = default_decider_class,
                default_decider_kwargs = default_decider_kwargs)
      
        pl.add_task(X_train, y_train, num_transformers=n_trees)

        y_hat = pl.predict(X_test, task_id=0)

        acc[i] = np.sum(y_test == y_hat) / n_samples_test

    return np.mean(acc)

def main():
    
    RF_acc = test_RF(sparseparity, 1000, 1000, 3, 10, 2, 10)
    print(RF_acc)


if __name__ == "__main__":
    main()

