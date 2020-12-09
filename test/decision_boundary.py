from rerf.rerfClassifier import rerfClassifier

import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.ensemble import RandomForestClassifier

from proglearn.progressive_learner import ClassificationProgressiveLearner
from proglearn.voters import TreeClassificationVoter
from proglearn.transformers import TreeClassificationTransformer
from proglearn.transformers import ObliqueTreeClassificationTransformer
from proglearn.deciders import SimpleArgmaxAverage

of1_kwargs = {"kwargs": {"feature_combinations" : 1.5, "density": 0.75}}
#of2_kwargs = {"kwargs": {"feature_combinations" : 1, "density": 0.75}}
NT = 10

pl1 = ClassificationProgressiveLearner(
        default_transformer_class = ObliqueTreeClassificationTransformer,
        default_transformer_kwargs = of1_kwargs,
        default_voter_class = TreeClassificationVoter,
        default_voter_kwargs = {},
        default_decider_class = SimpleArgmaxAverage,
        default_decider_kwargs = {"classes" : np.arange(2)}
        )

"""
pl2 = ClassificationProgressiveLearner(
        default_transformer_class = ObliqueTreeClassificationTransformer,
        default_transformer_kwargs = of2_kwargs,
        default_voter_class = TreeClassificationVoter,
        default_voter_kwargs = {},
        default_decider_class = SimpleArgmaxAverage,
        default_decider_kwargs = {"classes" : np.arange(2)}
        )
"""
h = .1  # step size in the mesh

names = ["RF", "RerF", "Proglearn-SPORF"]

classifiers = [
    RandomForestClassifier(max_depth=5, n_estimators=NT, max_features=1),
    rerfClassifier(n_estimators = NT, feature_combinations=1.5, max_features=2),
    of1_kwargs]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(15, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    print(len(y_train), len(y_test))

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
       
        print(name)

        if "Proglearn" in name:
            
            kwargs = clf
            clf = ClassificationProgressiveLearner(
                default_transformer_class = ObliqueTreeClassificationTransformer,
                default_transformer_kwargs = kwargs,
                default_voter_class = TreeClassificationVoter,
                default_voter_kwargs = {},
                default_decider_class = SimpleArgmaxAverage,
                default_decider_kwargs = {"classes" : np.arange(2)}
            )
            
            clf.add_task(X_train, y_train, num_transformers=NT)
            y_hat = clf.predict(X_test, task_id=0)
            score = np.sum(y_hat == y_test) / len(y_test)

        else:
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        elif "Proglearn" in name:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()], task_id=0)[:, 1]
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1
    
plt.tight_layout()
plt.savefig("decision_boundary")
