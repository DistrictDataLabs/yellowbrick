# tests.test_classifier.test_classification_report
# Tests for the classification report visualizer
#
# Author:  Rebecca Bilbro
# Author:  Benjamin Bengfort
# Created: Sun Mar 18 16:57:27 2018 -0400
#
# Copyright (C) 208 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_classification_report.py [] benjamin@bengfort.com $

"""
Tests for the classification report visualizer
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest
import yellowbrick as yb
import matplotlib.pyplot as plt

from yellowbrick.datasets import load_occupancy
from yellowbrick.classifier.classification_report import *

from pytest import approx
from unittest.mock import patch
from tests.base import VisualTestCase

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LassoCV, LogisticRegression

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
##  Test for Classification Report
##########################################################################


@pytest.mark.usefixtures("binary", "multiclass")
class TestClassificationReport(VisualTestCase):
    """
    ClassificationReport visualizer tests
    """

    def test_binary_class_report(self):
        """
        Correctly generates a report for binary classification with LinearSVC
        """
        _, ax = plt.subplots()

        viz = ClassificationReport(LinearSVC(random_state=42), ax=ax)
        viz.fit(self.binary.X.train, self.binary.y.train)
        viz.score(self.binary.X.test, self.binary.y.test)

        self.assert_images_similar(viz, tol=40)

        assert viz.scores_ == {
            "precision": {0: approx(0.7446808), 1: approx(0.8490566)},
            "recall": {0: approx(0.8139534), 1: approx(0.7894736)},
            "f1": {0: approx(0.7777777), 1: approx(0.8181818)},
        }

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_multiclass_class_report(self):
        """
        Correctly generates report for multi-class with LogisticRegression
        """
        _, ax = plt.subplots()

        viz = ClassificationReport(LogisticRegression(random_state=12), ax=ax)
        viz.fit(self.multiclass.X.train, self.multiclass.y.train)
        viz.score(self.multiclass.X.test, self.multiclass.y.test)

        self.assert_images_similar(viz, tol=11.0)

        assert viz.scores_ == {
            "precision": {
                0: 0.75,
                1: 0.47368421052631576,
                2: 0.45,
                3: 0.375,
                4: 0.5,
                5: 0.5294117647058824,
            },
            "recall": {
                0: 0.47368421052631576,
                1: 0.5625,
                2: 0.6428571428571429,
                3: 0.3157894736842105,
                4: 0.5,
                5: 0.5625,
            },
            "f1": {
                0: 0.5806451612903226,
                1: 0.5142857142857142,
                2: 0.5294117647058824,
                3: 0.34285714285714286,
                4: 0.5,
                5: 0.5454545454545455,
            },
        }

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_integration(self):
        """
        Test with Pandas DataFrame and Series input
        """
        _, ax = plt.subplots()

        # Load the occupancy dataset from fixtures
        data = load_occupancy(return_dataset=True)
        X, y = data.to_pandas()

        # Create train/test splits
        splits = tts(X, y, test_size=0.2, random_state=4512)
        X_train, X_test, y_train, y_test = splits

        classes = ["unoccupied", "occupied"]

        # Create classification report
        model = GaussianNB()
        viz = ClassificationReport(model, ax=ax, classes=classes)
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)

        self.assert_images_similar(viz, tol=5.0)

        # Ensure correct classification scores under the hood!
        assert viz.scores_ == {
            "precision": {
                "unoccupied": 0.999347471451876,
                "occupied": 0.8825214899713467,
            },
            "recall": {
                "unoccupied": 0.9613935969868174,
                "occupied": 0.9978401727861771,
            },
            "f1": {"unoccupied": 0.9800031994880819, "occupied": 0.9366447034972124},
        }

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_numpy_integration(self):
        """
        Test with NumPy arrays
        """
        _, ax = plt.subplots()

        # Load the occupancy dataset from fixtures
        data = load_occupancy(return_dataset=True)
        X, y = data.to_numpy()

        # Create train/test splits
        splits = tts(X, y, test_size=0.2, random_state=4512)
        X_train, X_test, y_train, y_test = splits

        classes = ["unoccupied", "occupied"]

        # Create classification report
        model = GaussianNB()
        viz = ClassificationReport(model, ax=ax, classes=classes)
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)

        self.assert_images_similar(viz, tol=5.0)

        # Ensure correct classification scores under the hood!
        assert viz.scores_ == {
            "precision": {
                "unoccupied": 0.999347471451876,
                "occupied": 0.8825214899713467,
            },
            "recall": {
                "unoccupied": 0.9613935969868174,
                "occupied": 0.9978401727861771,
            },
            "f1": {"unoccupied": 0.9800031994880819, "occupied": 0.9366447034972124},
        }

    def test_quick_method(self):
        """
        Test the quick method with a random dataset
        """
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=8,
            n_redundant=8,
            n_classes=2,
            n_clusters_per_class=4,
            random_state=27,
        )

        # Create train/test splits
        splits = tts(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = splits
        
        _, ax = plt.subplots()
        model = DecisionTreeClassifier(random_state=19)
        visualizer = classification_report(
            model, X_train, y_train, X_test, y_test, ax=ax, show=False
        )

        assert isinstance(visualizer, ClassificationReport)
        self.assert_images_similar(visualizer, tol=12)

    def test_isclassifier(self):
        """
        Assert that only classifiers can be used with the visualizer.
        """

        message = (
            "This estimator is not a classifier; "
            "try a regression or clustering score visualizer instead!"
        )

        with pytest.raises(yb.exceptions.YellowbrickError, match=message):
            ClassificationReport(LassoCV())

    def test_support_count_class_report(self):
        """
        Correctly generates a report showing support as a raw count
        """
        _, ax = plt.subplots()

        viz = ClassificationReport(LinearSVC(random_state=42), ax=ax, support="count")
        viz.fit(self.binary.X.train, self.binary.y.train)
        viz.score(self.binary.X.test, self.binary.y.test)

        self.assert_images_similar(viz, tol=40)

        assert viz.scores_ == {
            "precision": {0: approx(0.7446808), 1: approx(0.8490566)},
            "recall": {0: approx(0.8139534), 1: approx(0.7894736)},
            "f1": {0: approx(0.7777777), 1: approx(0.8181818)},
            "support": {0: approx(0.42999999999999999), 1: approx(0.56999999999999995)},
        }

    def test_support_percent_class_report(self):
        """
        Correctly generates a report showing support as a percent
        """
        _, ax = plt.subplots()

        viz = ClassificationReport(LinearSVC(random_state=42), ax=ax, support="percent")
        viz.fit(self.binary.X.train, self.binary.y.train)
        viz.score(self.binary.X.test, self.binary.y.test)

        self.assert_images_similar(viz, tol=40)

        assert viz.scores_ == {
            "precision": {0: approx(0.7446808), 1: approx(0.8490566)},
            "recall": {0: approx(0.8139534), 1: approx(0.7894736)},
            "f1": {0: approx(0.7777777), 1: approx(0.8181818)},
            "support": {0: approx(0.42999999999999999), 1: approx(0.56999999999999995)},
        }

    def test_invalid_support(self):
        """
        Ensure that bad support arguments raise exception
        """
        with pytest.raises(
            YellowbrickValueError,
            match="'foo' is an invalid argument for support, use None, "
            "True, False, 'percent', or 'count'",
        ):
            ClassificationReport(LinearSVC(), support="foo")

    def test_score_returns_score(self):
        """
        Test that ClassificationReport score() returns a score between 0 and 1
        """
        viz = ClassificationReport(LinearSVC(random_state=42))

        viz.fit(self.binary.X.train, self.binary.y.train)
        s = viz.score(self.binary.X.test, self.binary.y.test)

        assert 0 <= s <= 1

    def test_with_fitted(self):
        """
        Test that visualizer properly handles an already-fitted model
        """
        X, y = load_occupancy(return_dataset=True).to_numpy()

        model = LinearSVC().fit(X, y)
        classes = ["unoccupied", "occupied"]

        with patch.object(model, "fit") as mockfit:
            oz = ClassificationReport(model, classes=classes)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with patch.object(model, "fit") as mockfit:
            oz = ClassificationReport(model, classes=classes, is_fitted=True)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with patch.object(model, "fit") as mockfit:
            oz = ClassificationReport(model, classes=classes, is_fitted=False)
            oz.fit(X, y)
            mockfit.assert_called_once_with(X, y)
