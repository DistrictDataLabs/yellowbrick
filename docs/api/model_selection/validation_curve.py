#!/usr/bin/env python3
# validation_curve.py

"""
Generates the validation curve visualizations for the documentation
"""

##########################################################################
## Imports
##########################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from yellowbrick.model_selection import ValidationCurve

FIXTURES = os.path.join("..", "..", "..", "examples", "data")


##########################################################################
## Helper Methods
##########################################################################

def validation_curve_sklearn_example(path="images/validation_curve_sklearn_example.png"):
    digits = load_digits()
    X, y = digits.data, digits.target

    _, ax = plt.subplots()

    param_range = np.logspace(-6, -1, 5)
    oz = ValidationCurve(
        SVC(), ax=ax, param_name="gamma", param_range=param_range,
        logx=True, cv=10, scoring="accuracy", n_jobs=4
    )
    oz.fit(X, y)
    oz.poof(outpath=path)


def validation_curve_classifier(path="images/validation_curve_classifier.png"):
    data = pd.read_csv(os.path.join(FIXTURES, "game", "game.csv"))

    target = "outcome"
    features = [col for col in data.columns if col != target]

    X = pd.get_dummies(data[features])
    y = data[target]

    _, ax = plt.subplots()
    cv = StratifiedKFold(12)
    param_range = np.logspace(-6, -1, 12)

    oz = ValidationCurve(
        SVC(), ax=ax, param_name="gamma", param_range=param_range,
        logx=True, cv=cv, scoring="f1_weighted", n_jobs=8,
    )
    oz.fit(X, y)
    oz.poof(outpath=path)


def validation_curve_classifier_alt(path="images/validation_curve_classifier_alt.png"):
    data = pd.read_csv(os.path.join(FIXTURES, "game", "game.csv"))

    target = "outcome"
    features = [col for col in data.columns if col != target]

    X = pd.get_dummies(data[features])
    y = data[target]

    _, ax = plt.subplots()
    cv = StratifiedKFold(4)
    param_range = np.arange(3, 20, 2)

    oz = ValidationCurve(
        KNeighborsClassifier(), ax=ax, param_name="n_neighbors",
        param_range=param_range, cv=cv, scoring="f1_weighted", n_jobs=8,
    )
    oz.fit(X, y)
    oz.poof(outpath=path)


def validation_curve_regressor(path="images/validation_curve_regressor.png"):

    data = pd.read_csv(os.path.join(FIXTURES, "energy", "energy.csv"))

    targets = ["heating load", "cooling load"]
    features = [col for col in data.columns if col not in targets]

    X = data[features]
    y = data[targets[1]]

    _, ax = plt.subplots()
    param_range = np.arange(1, 11)

    oz = ValidationCurve(
        DecisionTreeRegressor(), ax=ax, param_name="max_depth",
        param_range=param_range, cv=10, scoring="r2", n_jobs=8,
    )
    oz.fit(X, y)
    oz.poof(outpath=path)


##########################################################################
## Main Method
##########################################################################

if __name__ == '__main__':
    validation_curve_sklearn_example()
    # validation_curve_classifier()
    validation_curve_classifier_alt()
    validation_curve_regressor()
