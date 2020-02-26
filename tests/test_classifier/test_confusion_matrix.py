# tests.test_classifier.test_confusion_matrix
# Tests for the confusion matrix visualizer
#
# Aithor:  Neal Humphrey
# Author:  Benjamin Bengfort
# Created: Tue May 03 11:05:11 2017 -0700
#
# Copyright (C) 2017 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_confusion_matrix.py [] benjamin@bengfort.com $

"""
Tests for the confusion matrix visualizer
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest
import yellowbrick as yb
import numpy.testing as npt
import matplotlib.pyplot as plt

from yellowbrick.exceptions import ModelError
from yellowbrick.datasets import load_occupancy
from yellowbrick.classifier.confusion_matrix import *

from unittest.mock import patch
from tests.fixtures import Dataset, Split
from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.datasets import load_digits, make_classification

try:
    import pandas as pd
except ImportError:
    pd = None

##########################################################################
## Fixtures
##########################################################################


@pytest.fixture(scope="class")
def digits(request):
    """
    Creates a fixture of train and test splits for the sklearn digits dataset
    For ease of use returns a Dataset named tuple composed of two Split tuples.
    """
    data = load_digits()
    X_train, X_test, y_train, y_test = tts(
        data.data, data.target, test_size=0.2, random_state=11
    )

    # Set a class attribute for digits
    request.cls.digits = Dataset(Split(X_train, X_test), Split(y_train, y_test))


##########################################################################
## Test Cases
##########################################################################


@pytest.mark.usefixtures("digits")
class TestConfusionMatrix(VisualTestCase):
    """
    Test ConfusionMatrix visualizer
    """

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_confusion_matrix(self):
        """
        Integration test on digits dataset with LogisticRegression
        """
        _, ax = plt.subplots()

        model = LogisticRegression(random_state=93)
        cm = ConfusionMatrix(model, ax=ax, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        cm.fit(self.digits.X.train, self.digits.y.train)
        cm.score(self.digits.X.test, self.digits.y.test)

        self.assert_images_similar(cm, tol=10)

        # Ensure correct confusion matrix under the hood
        npt.assert_array_equal(
            cm.confusion_matrix_,
            np.array(
                [
                    [38, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 35, 0, 0, 0, 0, 0, 0, 2, 0],
                    [0, 0, 39, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 38, 0, 0, 0, 0, 3, 0],
                    [0, 0, 0, 0, 40, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 27, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 29, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 35, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0, 1, 31, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 36],
                ]
            ),
        )

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_no_classes_provided(self):
        """
        Integration test on digits dataset with GaussianNB, no classes
        """
        _, ax = plt.subplots()

        model = GaussianNB()
        cm = ConfusionMatrix(model, ax=ax, classes=None)
        cm.fit(self.digits.X.train, self.digits.y.train)
        cm.score(self.digits.X.test, self.digits.y.test)

        self.assert_images_similar(cm, tol=10)

        # Ensure correct confusion matrix under the hood
        npt.assert_array_equal(
            cm.confusion_matrix_,
            np.array(
                [
                    [36, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 31, 0, 0, 0, 0, 0, 1, 3, 2],
                    [0, 1, 34, 0, 0, 0, 0, 0, 4, 0],
                    [0, 1, 0, 33, 0, 2, 0, 2, 3, 0],
                    [0, 0, 0, 0, 36, 0, 0, 5, 0, 0],
                    [0, 0, 0, 0, 0, 27, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0, 28, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 36, 0, 0],
                    [0, 3, 0, 1, 0, 1, 0, 4, 25, 0],
                    [1, 2, 0, 0, 1, 0, 0, 8, 3, 22],
                ]
            ),
        )

    def test_fontsize(self):
        """
        Test confusion matrix with smaller fontsize on digits dataset with SVC
        """
        _, ax = plt.subplots()

        model = SVC(random_state=93)
        cm = ConfusionMatrix(model, ax=ax, fontsize=8)

        cm.fit(self.digits.X.train, self.digits.y.train)
        cm.score(self.digits.X.test, self.digits.y.test)

        self.assert_images_similar(cm, tol=10)

    def test_percent_mode(self):
        """
        Test confusion matrix in percent mode on digits dataset with SVC
        """
        _, ax = plt.subplots()

        model = SVC(random_state=93)
        cm = ConfusionMatrix(model, ax=ax, percent=True)

        cm.fit(self.digits.X.train, self.digits.y.train)
        cm.score(self.digits.X.test, self.digits.y.test)

        self.assert_images_similar(cm, tol=10)

        # Ensure correct confusion matrix under the hood
        npt.assert_array_equal(
            cm.confusion_matrix_,
            np.array(
                [
                    [38, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 37, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 39, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 38, 0, 1, 0, 1, 1, 0],
                    [0, 0, 0, 0, 41, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 27, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 30, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 35, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 34, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 36],
                ]
            ),
        )

    @pytest.mark.xfail(reason="class filtering is not currently supported")
    def test_class_filter_eg_zoom_in(self):
        """
        Test filtering classes zooms in on the confusion matrix.
        """
        _, ax = plt.subplots()

        model = LogisticRegression(random_state=93)
        cm = ConfusionMatrix(model, ax=ax, classes=[0, 1, 2])
        cm.fit(self.digits.X.train, self.digits.y.train)
        cm.score(self.digits.X.test, self.digits.y.test)

        self.assert_images_similar(cm, tol=10)

        # Ensure correct confusion matrix under the hood
        npt.assert_array_equal(
            cm.confusion_matrix_, np.array([[38, 0, 0], [0, 35, 0], [0, 0, 39]])
        )

    def test_extra_classes(self):
        """
        Assert that any extra classes raise an exception
        """
        model = LogisticRegression(random_state=93)
        cm = ConfusionMatrix(model, classes=[0, 1, 2, 11])

        with pytest.raises(ModelError, match="could not decode"):
            cm.fit(self.digits.X.train, self.digits.y.train)

    def test_defined_mapping(self):
        """
        Test mapping as label encoder to define tick labels
        """
        _, ax = plt.subplots()

        model = LogisticRegression(random_state=93)
        classes = np.array(
            [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
            ]
        )
        mapping = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
        }
        cm = ConfusionMatrix(model, ax=ax, encoder=mapping)
        cm.fit(self.digits.X.train, self.digits.y.train)
        cm.score(self.digits.X.test, self.digits.y.test)

        xlabels = np.array([l.get_text() for l in ax.get_xticklabels()])
        npt.assert_array_equal(xlabels, classes)

        ylabels = [l.get_text() for l in ax.get_yticklabels()]
        ylabels.reverse()
        ylabels = np.asarray(ylabels)
        npt.assert_array_equal(ylabels, classes)

    def test_inverse_mapping(self):
        """
        Test LabelEncoder as label encoder to define tick labels
        """
        _, ax = plt.subplots()

        model = LogisticRegression(random_state=93)
        le = LabelEncoder()
        le.fit(
            [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
            ]
        )

        cm = ConfusionMatrix(model, ax=ax, encoder=le)
        cm.fit(self.digits.X.train, self.digits.y.train)
        cm.score(self.digits.X.test, self.digits.y.test)

        xlabels = np.array([l.get_text() for l in ax.get_xticklabels()])
        npt.assert_array_equal(xlabels, le.classes_)

        ylabels = [l.get_text() for l in ax.get_yticklabels()]
        ylabels.reverse()
        ylabels = np.asarray(ylabels)
        npt.assert_array_equal(ylabels, le.classes_)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_integration(self):
        """
        Test with Pandas DataFrame and Series input
        """
        _, ax = plt.subplots()

        # Load the occupancy dataset from fixtures
        X, y = load_occupancy(return_dataset=True).to_pandas()

        # Create train/test splits
        splits = tts(X, y, test_size=0.2, random_state=8873)
        X_train, X_test, y_train, y_test = splits

        # Create confusion matrix
        model = GaussianNB()
        cm = ConfusionMatrix(model, ax=ax, classes=None)
        cm.fit(X_train, y_train)
        cm.score(X_test, y_test)

        self.assert_images_similar(cm, tol=0.1)

        # Ensure correct confusion matrix under the hood
        npt.assert_array_equal(cm.confusion_matrix_, np.array([[3012, 114], [1, 985]]))

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
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

        _, ax = plt.subplots()
        model = DecisionTreeClassifier(random_state=25)

        # compare the images
        visualizer = confusion_matrix(model, X, y, ax=ax, show=False)
        self.assert_images_similar(visualizer)

    def test_isclassifier(self):
        """
        Assert that only classifiers can be used with the visualizer.
        """
        model = PassiveAggressiveRegressor()
        message = (
            "This estimator is not a classifier; "
            "try a regression or clustering score visualizer instead!"
        )

        with pytest.raises(yb.exceptions.YellowbrickError, match=message):
            ConfusionMatrix(model)

    def test_score_returns_score(self):
        """
        Test that ConfusionMatrix score() returns a score between 0 and 1
        """
        # Load the occupancy dataset from fixtures
        X, y = load_occupancy(return_dataset=True).to_numpy()
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)

        # Create and fit the visualizer
        visualizer = ConfusionMatrix(LogisticRegression())
        visualizer.fit(X_train, y_train)

        # Score the visualizer
        s = visualizer.score(X_test, y_test)

        assert 0 <= s <= 1

    def test_with_fitted(self):
        """
        Test that visualizer properly handles an already-fitted model
        """
        X, y = load_occupancy(return_dataset=True).to_numpy()

        model = LogisticRegression().fit(X, y)
        classes = ["unoccupied", "occupied"]

        with patch.object(model, "fit") as mockfit:
            oz = ConfusionMatrix(model, classes=classes)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with patch.object(model, "fit") as mockfit:
            oz = ConfusionMatrix(model, classes=classes, is_fitted=True)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with patch.object(model, "fit") as mockfit:
            oz = ConfusionMatrix(model, classes=classes, is_fitted=False)
            oz.fit(X, y)
            mockfit.assert_called_once_with(X, y)
