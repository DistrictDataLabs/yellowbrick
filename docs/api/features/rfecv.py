#!/usr/bin/env python3
# Generates RFECV visualizations for the documentation

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from yellowbrick.features import RFECV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

CWD    = os.path.dirname(__file__)
DATA   = os.path.join(CWD, "..", "..", "..", "examples", "data")
IMAGES = os.path.join(CWD, "images")


def rfecv_sklearn_example(image="rfecv_sklearn_example.png"):
    X, y = make_classification(
        n_samples=1000, n_features=25, n_informative=3, n_redundant=2,
        n_repeated=0, n_classes=8, n_clusters_per_class=1, random_state=0
    )

    _, ax = plt.subplots()

    oz = RFECV(SVC(kernel='linear', C=1), ax=ax)
    oz.fit(X, y)
    oz.poof(outpath=os.path.join(IMAGES, image))


def rfecv_credit_example(image="rfecv_credit.png"):
    data = pd.read_csv(os.path.join(DATA, "credit", "credit.csv"))

    target = "default"
    features = [col for col in data.columns if col != target]

    X = data[features]
    y = data[target]

    _, ax = plt.subplots()
    cv = StratifiedKFold(5)
    oz = RFECV(RandomForestClassifier(), ax=ax, cv=cv, scoring='f1_weighted')
    oz.fit(X, y)
    oz.poof(outpath=os.path.join(IMAGES, image))


if __name__ == '__main__':
    rfecv_sklearn_example()
    rfecv_credit_example()
