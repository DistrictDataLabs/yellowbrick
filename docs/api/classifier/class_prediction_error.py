# class_prediction_error.py

"""
Creates the visualizations for the class_prediction_error.rst documentation
"""

##########################################################################
## Imports
##########################################################################

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts

from yellowbrick.classifier import ClassPredictionError


def make_fruit_dataset():
    X, y = make_classification(
        n_samples=1000, n_classes=5, n_informative=3, n_clusters_per_class=1
    )

    classes = ['apple', 'kiwi', 'pear', 'banana', 'orange']
    return tts(X, y, test_size=0.20, random_state=42), classes


def load_credit_dataset():
    data = pd.read_csv("../../../examples/data/credit/credit.csv")
    target = "default"
    features = list(data.columns)
    features.remove(target)

    X = data[features]
    y = data[target]

    classes = ["default", "current"]
    return tts(X, y, test_size=0.2, random_state=53), classes


def make_cb_pred_error(dataset="fruit", path=None, clf=None):
    clf = clf or RandomForestClassifier()

    loader = {
        'fruit': make_fruit_dataset,
        'credit': load_credit_dataset,
    }[dataset]

    (X_train, X_test, y_train, y_test), classes = loader()

    _, ax = plt.subplots()
    viz =  ClassPredictionError(clf, ax=ax, classes=classes)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)

    return viz.poof(outpath=path)


if __name__ == '__main__':
    make_cb_pred_error("fruit", "images/class_prediction_error.png")
    make_cb_pred_error("credit", "images/class_prediction_error_credit.png")
