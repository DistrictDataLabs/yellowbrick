# tests.test_classifier.test_class_prediction_error
# Testing for the ClassPredictionError visualizer
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Larry Gray
# Created:  Tue May 23 13:41:55 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_rocauc.py [] benjamin@bengfort.com $

"""
Testing for the ClassPredictionError visualizer
"""

##########################################################################
## Imports
##########################################################################

import pytest
import matplotlib.pyplot as plt

from yellowbrick.classifier.class_balance import *
from yellowbrick.exceptions import ModelError

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_multilabel_classification, make_classification

from tests.base import VisualTestCase

##########################################################################
## Data
##########################################################################

X, y = make_classification(
    n_classes=4, n_informative=3, n_clusters_per_class=1, random_state=42
)

##########################################################################
##  Tests
##########################################################################


class ClassPredictionErrorTests(VisualTestCase):

    def test_integration_class_prediction_error_(self):
        """
        Assert no errors occur during class prediction error integration
        """
        model = LinearSVC()
        model.fit(X, y)
        visualizer = ClassPredictionError(model, classes=["A", "B", "C", "D"])
        visualizer.score(X, y)
        self.assert_images_similar(visualizer)

    def test_class_prediction_error_quickmethod(self):
        """
        Test the ClassPreditionError quickmethod
        """
        fig = plt.figure()
        ax = fig.add_subplot()

        clf = LinearSVC(random_state=42)
        g = class_prediction_error(clf, X, y, ax)

        self.assert_images_similar(ax=g)

    def test_classes_greater_than_indices(self):
        """
        Assert error when y and y_pred contain zero values for
        one of the specified classess
        """
        model = LinearSVC()
        model.fit(X, y)
        with self.assertRaises(ModelError):
            visualizer = ClassPredictionError(
                model, classes=["A", "B", "C", "D", "E"]
            )
            visualizer.score(X, y)

    def test_classes_less_than_indices(self):
        """
        Assert error when there is an attempt to filter classes
        """
        model = LinearSVC()
        model.fit(X, y)
        with self.assertRaises(NotImplementedError):
            visualizer = ClassPredictionError(model, classes=["A"])
            visualizer.score(X, y)

    @pytest.mark.skip(reason="not implemented yet")
    def test_no_classes_provided(self):
        """
        Assert no errors when no classes are provided
        """
        pass

    def test_class_type(self):
        """
        Test class must be either binary or multiclass type
        """
        X, y = make_multilabel_classification()
        model = RandomForestClassifier()
        model.fit(X, y)
        with self.assertRaises(YellowbrickValueError):
            visualizer = ClassPredictionError(model)
            visualizer.score(X, y)
