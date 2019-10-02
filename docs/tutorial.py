#!/usr/bin/env python
# Generate the classification report images for the tutorial

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)

from yellowbrick.datasets import load_mushroom
from yellowbrick.classifier import ClassificationReport


ESTIMATORS = {
    "SVC": {"model": SVC(gamma="auto"), "path": "images/tutorial/modelselect_svc.png"},
    "NuSVC": {
        "model": NuSVC(gamma="auto"),
        "path": "images/tutorial/modelselect_nu_svc.png",
    },
    "LinearSVC": {
        "model": LinearSVC(),
        "path": "images/tutorial/modelselect_linear_svc.png",
    },
    "SGD": {
        "model": SGDClassifier(max_iter=100, tol=1e-3),
        "path": "images/tutorial/modelselect_sgd_classifier.png",
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "path": "images/tutorial/modelselect_kneighbors_classifier.png",
    },
    "LR": {
        "model": LogisticRegression(solver="lbfgs"),
        "path": "images/tutorial/modelselect_logistic_regression.png",
    },
    "LRCV": {
        "model": LogisticRegressionCV(cv=3),
        "path": "images/tutorial/modelselect_logistic_regression_cv.png",
    },
    "Bags": {
        "model": BaggingClassifier(),
        "path": "images/tutorial/modelselect_bagging_classifier.png",
    },
    "XTrees": {
        "model": ExtraTreesClassifier(n_estimators=100),
        "path": "images/tutorial/modelselect_extra_trees_classifier.png",
    },
    "RF": {
        "model": RandomForestClassifier(n_estimators=100),
        "path": "images/tutorial/modelselect_random_forest_classifier.png",
    },
}


def visualize_model(X, y, estimator, path, **kwargs):
    """
    Test various estimators.
    """
    y = LabelEncoder().fit_transform(y)
    model = Pipeline([("one_hot_encoder", OneHotEncoder()), ("estimator", estimator)])

    _, ax = plt.subplots()

    # Instantiate the classification model and visualizer
    visualizer = ClassificationReport(
        model,
        classes=["edible", "poisonous"],
        cmap="YlGn",
        size=(600, 360),
        ax=ax,
        **kwargs
    )
    visualizer.fit(X, y)
    visualizer.score(X, y)
    visualizer.show(outpath=path)


if __name__ == "__main__":
    X, y = load_mushroom()

    for clf in ESTIMATORS.values():
        visualize_model(X, y, clf["model"], clf["path"])
