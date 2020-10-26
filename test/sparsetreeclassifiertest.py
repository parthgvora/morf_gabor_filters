import numpy as np
from sklearn.datasets import *
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import ObliqueForestClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

def test(clf, dataset):
    res = cross_val_score(clf, dataset.data, dataset.target, cv=10)
    return np.mean(res)
    

def test_RF():
    iris_score = test(RandomForestClassifier(), load_iris())
    digits_score = test(RandomForestClassifier(), load_digits())
    wine_score = test(RandomForestClassifier(), load_wine())
    breast_cancer_score = test(RandomForestClassifier(), load_breast_cancer())
    return [iris_score, digits_score, wine_score, breast_cancer_score]

def test_OF():
    kwargs = {
            "feature_combinations":1.5,
            "density":0.2
    }

    iris_score = test(ObliqueForestClassifier(**kwargs), load_iris())
    digits_score = test(ObliqueForestClassifier(**kwargs), load_digits())
    wine_score = test(ObliqueForestClassifier(**kwargs), load_wine())
    breast_cancer_score = test(ObliqueForestClassifier(**kwargs), load_breast_cancer())
    return [iris_score, digits_score, wine_score, breast_cancer_score]

print("Random Forest CV scores")
RF_scores = test_RF()

print("Oblique Forest CV scores")
OF_scores = test_OF()

X1 = [10, 15, 20, 25]
X2 = [11, 16, 21, 26]

plt.figure()
plt.title("Comparison of RF and sklearn-SPORF on toy datasets")
plt.bar(X1, RF_scores, color="red")
plt.bar(X2, OF_scores, color="blue")
plt.xlabel("Dataset")
plt.xticks(X1, ["iris", "digits", "wine", "breast cancer"])
plt.ylabel("10-fold cross validation score")
plt.legend(["Random Forest", "SPORF"], loc=4)
plt.savefig("sklearn_dataset_scores")


