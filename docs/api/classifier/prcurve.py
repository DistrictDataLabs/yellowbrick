#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from yellowbrick.classifier import PrecisionRecallCurve

from sklearn.datasets import load_iris
from yellowbrick.datasets import load_occupancy

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split


def draw_binary(outpath=None):
    data = load_occupancy()
    X = data[["temperature", "relative_humidity", "light", "C02", "humidity"]].copy()
    X = X.view((float, len(X.dtype.names)))
    y = data["occupancy"]

    n_samples, n_features = X.shape
    random_state = np.random.RandomState(42)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, shuffle=True)

    oz = PrecisionRecallCurve(LinearSVC())
    oz.fit(X_train, y_train)
    oz.score(X_test, y_test)
    oz.poof(outpath=outpath)


def draw_multiclass(outpath=None, simple=True):

    data = load_iris()
    X = data.data
    y = data.target

    n_samples, n_features = X.shape
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, shuffle=True)


    if simple:
        oz = PrecisionRecallCurve(GaussianNB())
    else:
        oz = PrecisionRecallCurve(RidgeClassifier(), per_class=True, iso_f1_curves=True, fill_area=False, micro=False)

    oz.fit(X_train, y_train)
    oz.score(X_test, y_test)
    oz.poof(outpath=outpath)



if __name__ == '__main__':
    # draw_binary()
    # draw_multiclass(simple=True)
    draw_multiclass(simple=False)
