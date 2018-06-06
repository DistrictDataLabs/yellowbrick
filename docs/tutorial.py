#!/usr/bin/env python
# Generate the classification report images for the tutorial

import os
import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.classifier import ClassificationReport

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier


DATA = os.path.join(
    os.path.dirname(__file__),  "..", "examples", "data", "mushroom", "mushroom.csv"
)


ESTIMATORS = {
    LinearSVC: "images/tutorial/modelselect_linear_svc.png",
    NuSVC: "images/tutorial/modelselect_nu_svc.png",
    SVC: "images/tutorial/modelselect_svc.png",
    SGDClassifier: "images/tutorial/modelselect_sgd_classifier.png",
    KNeighborsClassifier: "images/tutorial/modelselect_kneighbors_classifier.png",
    LogisticRegressionCV: "images/tutorial/modelselect_logistic_regression_cv.png",
    LogisticRegression: "images/tutorial/modelselect_logistic_regression.png",
    BaggingClassifier: "images/tutorial/modelselect_bagging_classifier.png",
    ExtraTreesClassifier: "images/tutorial/modelselect_extra_trees_classifier.png",
    RandomForestClassifier: "images/tutorial/modelselect_random_forest_classifier.png",
}



class EncodeCategorical(BaseEstimator, TransformerMixin):
    """
    Encodes a specified list of columns or all columns if None.
    """

    def __init__(self, columns=None):
        self.columns  = [col for col in columns]
        self.encoders = None

    def fit(self, data, target=None):
        """
        Expects a data frame with named columns to encode.
        """
        # Encode all columns if columns is None
        if self.columns is None:
            self.columns = data.columns

        # Fit a label encoder for each column in the data frame
        self.encoders = {
            column: LabelEncoder().fit(data[column])
            for column in self.columns
        }
        return self

    def transform(self, data):
        """
        Uses the encoders to transform a data frame.
        """
        output = data.copy()
        for column, encoder in self.encoders.items():
            output[column] = encoder.transform(data[column])

        return output


def load_data(path=DATA):
    dataset  = pd.read_csv(path)
    features = ['shape', 'surface', 'color']
    target   = ['target']

    X = dataset[features]
    y = dataset[target]

    y = LabelEncoder().fit_transform(y.values.ravel())

    return X, y


def visual_model_selection(X, y, estimator, path):
    """
    Test various estimators.
    """
    model = Pipeline([
         ('label_encoding', EncodeCategorical(X.keys())),
         ('one_hot_encoder', OneHotEncoder()),
         ('estimator', estimator)
    ])

    _, ax = plt.subplots()

    # Instantiate the classification model and visualizer
    visualizer = ClassificationReport(model, ax=ax, classes=['edible', 'poisonous'])
    visualizer.fit(X, y)
    visualizer.score(X, y)
    visualizer.poof(outpath=path)


if __name__ == '__main__':
    X, y = load_data()

    for clf, path in ESTIMATORS.items():
        visual_model_selection(X, y, clf(), path)
