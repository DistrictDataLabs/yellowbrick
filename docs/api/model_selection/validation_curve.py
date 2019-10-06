#!/usr/bin/env python3
# validation_curve.py

"""
Generates the validation curve visualizations for the documentation
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from yellowbrick.datasets import load_game
from yellowbrick.model_selection import ValidationCurve


##########################################################################
## Helper Methods
##########################################################################


def validation_curve_sklearn_example(
    path="images/validation_curve_sklearn_example.png"
):
    digits = load_digits()
    X, y = digits.data, digits.target

    _, ax = plt.subplots()

    param_range = np.logspace(-6, -1, 5)
    oz = ValidationCurve(
        SVC(),
        ax=ax,
        param_name="gamma",
        param_range=param_range,
        logx=True,
        cv=10,
        scoring="accuracy",
        n_jobs=4,
    )
    oz.fit(X, y)
    oz.show(outpath=path)


def validation_curve_classifier_svc(path="images/validation_curve_classifier_svc.png"):
    X, y = load_game()
    X = OneHotEncoder().fit_transform(X)

    _, ax = plt.subplots()
    cv = StratifiedKFold(12)
    param_range = np.logspace(-6, -1, 12)

    print("warning: generating the SVC validation curve can take a very long time!")

    oz = ValidationCurve(
        SVC(),
        ax=ax,
        param_name="gamma",
        param_range=param_range,
        logx=True,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=8,
    )
    oz.fit(X, y)
    oz.show(outpath=path)


def validation_curve_classifier_knn(path="images/validation_curve_classifier_knn.png"):
    X, y = load_game()
    X = OneHotEncoder().fit_transform(X)

    _, ax = plt.subplots()
    cv = StratifiedKFold(4)
    param_range = np.arange(3, 20, 2)

    print("warning: generating the KNN validation curve can take a very long time!")

    oz = ValidationCurve(
        KNeighborsClassifier(),
        ax=ax,
        param_name="n_neighbors",
        param_range=param_range,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=8,
    )
    oz.fit(X, y)
    oz.show(outpath=path)


##########################################################################
## Main Method
##########################################################################

if __name__ == "__main__":
    # validation_curve_sklearn_example()
    validation_curve_classifier_svc()
    validation_curve_classifier_knn()
