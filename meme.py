import numpy as np
from numpy import random as rng
from sklearn.random_projection import SparseRandomProjection as srp
from sklearn.tree import DecisionTreeClassifier

import sys
sys.path.append("../")
from oblique_tree import ObliqueTreeClassifier

from sklearn.datasets import load_iris as load

data = load()

#clf = DecisionTreeClassifier()
clf = ObliqueTreeClassifier()
print(data.data.shape)

clf.fit(data.data, data.target)
print(clf.apply(data.data))

#preds = clf.predict(data.data)

#acc = sum(clf.predict(data.data) == data.target) / len(data.target)
#print(acc)
