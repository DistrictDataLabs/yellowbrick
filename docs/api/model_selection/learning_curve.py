#!/usr/bin/env python3
# learning_curve.py

"""
Generates the learning curve visualizations for the documentation
"""

##########################################################################
## Imports
##########################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.datasets import load_digits
from sklearn.linear_model import RidgeCV
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold

from yellowbrick.model_selection import LearningCurve


FIXTURES = os.path.join("..", "..", "..", "examples", "data")


##########################################################################
## Helper Methods
##########################################################################

def learning_curve_sklearn_example(path="images/learning_curve_sklearn_example.png"):
    digits = load_digits()
    X, y = digits.data, digits.target

    _, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(9,4))

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    oz = LearningCurve(GaussianNB(), ax=ax[0], cv=cv, n_jobs=4)
    oz.fit(X, y)
    oz.finalize()

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    oz = LearningCurve(SVC(gamma=0.001), ax=ax[1], cv=cv, n_jobs=4)
    oz.fit(X, y)
    oz.poof(outpath=path)


def learning_curve_classifier(path="images/learning_curve_classifier.png"):

    data = pd.read_csv(os.path.join(FIXTURES, "game", "game.csv"))

    target = "outcome"
    features = [col for col in data.columns if col != target]

    X = pd.get_dummies(data[features])
    y = data[target]

    _, ax = plt.subplots()
    cv = StratifiedKFold(12)
    sizes = np.linspace(0.3, 1.0, 10)

    oz = LearningCurve(
        MultinomialNB(), ax=ax, cv=cv, n_jobs=4,
        train_sizes=sizes, scoring='f1_weighted'
    )

    oz.fit(X, y)
    oz.poof(outpath=path)


def learning_curve_regressor(path="images/learning_curve_regressor.png"):

    data = pd.read_csv(os.path.join(FIXTURES, "energy", "energy.csv"))

    targets = ["heating load", "cooling load"]
    features = [col for col in data.columns if col not in targets]

    X = data[features]
    y = data[targets[0]]

    _, ax = plt.subplots()
    sizes = np.linspace(0.3, 1.0, 10)

    oz = LearningCurve(RidgeCV(), ax=ax, train_sizes=sizes, scoring='r2')
    oz.fit(X, y)
    oz.poof(outpath=path)


def learning_curve_clusterer(path="images/learning_curve_clusterer.png"):

    X, y = make_blobs(n_samples=1000, centers=5)

    _, ax = plt.subplots()
    sizes = np.linspace(0.3, 1.0, 10)

    oz = LearningCurve(
        KMeans(), ax=ax, train_sizes=sizes, scoring="adjusted_rand_score"
    )
    oz.fit(X, y)
    oz.poof(outpath=path)

##########################################################################
## Main Method
##########################################################################

if __name__ == '__main__':
    learning_curve_sklearn_example()
    learning_curve_classifier()
    learning_curve_regressor()
    learning_curve_clusterer()
