#!/usr/bin/env python3
# rfecv.py
# Generates RFECV visualizations for the documentation
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: rfecv.py [] $

import os
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from yellowbrick.model_selection import RFECV, rfecv
from yellowbrick.datasets import load_credit


CWD = os.path.dirname(__file__)
IMAGES = os.path.join(CWD, "images")


def rfecv_sklearn_example(image="rfecv_sklearn_example.png"):
    X, y = make_classification(
        n_samples=1000,
        n_features=25,
        n_informative=3,
        n_redundant=2,
        n_repeated=0,
        n_classes=8,
        n_clusters_per_class=1,
        random_state=0,
    )

    _, ax = plt.subplots()

    oz = RFECV(SVC(kernel="linear", C=1), ax=ax)
    oz.fit(X, y)
    oz.show(outpath=os.path.join(IMAGES, image))


def rfecv_credit_example(image="rfecv_credit.png"):
    X, y = load_credit()

    _, ax = plt.subplots()
    cv = StratifiedKFold(5)
    oz = RFECV(RandomForestClassifier(), ax=ax, cv=cv, scoring="f1_weighted")
    oz.fit(X, y)
    oz.show(outpath=os.path.join(IMAGES, image))


def rfecv_quick_method(image="rfecv_quick_method.png"):
    X, y = load_credit()

    _, ax = plt.subplots()
    cv = StratifiedKFold(5)
    visualizer = rfecv(RandomForestClassifier(), X=X, y=y, ax=ax, cv=cv, scoring='f1_weighted')
    visualizer.show(outpath=os.path.join(IMAGES, image))


##########################################################################
## Main Method
##########################################################################

if __name__ == "__main__":
    rfecv_sklearn_example()
    rfecv_credit_example()
    rfecv_quick_method()
