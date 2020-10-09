# tests.test_classifier.test_class_prediction_error
# Testing for the ClassPredictionError visualizer
#
# Author:   Benjamin Bengfort
# Author:   Rebecca Bilbro
# Author:   Larry Gray
# Created:  Tue May 23 13:41:55 2017 -0700
#
# Copyright (C) 2017 The scikit-yb developers
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

from yellowbrick.exceptions import ModelError
from yellowbrick.datasets import load_occupancy
from yellowbrick.classifier.class_prediction_error import *

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split as tts

from unittest.mock import patch
from tests.base import VisualTestCase


try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
##  Tests
##########################################################################


class TestClassPredictionError(VisualTestCase):
    """
    Test ClassPredictionError visualizer
    """

    @pytest.mark.filterwarnings("ignore:could not determine class_counts_")
    def test_numpy_integration(self):
        """
        Assert no errors during class prediction error integration with NumPy arrays
        """
        X, y = load_occupancy(return_dataset=True).to_numpy()

        classes = ["unoccupied", "occupied"]

        model = SVC(random_state=42)
        model.fit(X, y)
        visualizer = ClassPredictionError(model, classes=classes)
        visualizer.score(X, y)
        visualizer.finalize()

        # AppVeyor and Linux conda fail due to non-text-based differences
        # AppVeyor fails with RMS 13.161 - 13.289 (python - miniconda)
        self.assert_images_similar(visualizer, tol=12.5, windows_tol=13.3)

    @pytest.mark.filterwarnings("ignore:could not determine class_counts_")
    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_integration(self):
        """
        Assert no errors during class prediction error integration with Pandas
        """
        X, y = load_occupancy(return_dataset=True).to_pandas()
        classes = ["unoccupied", "occupied"]

        model = SVC(random_state=42)
        model.fit(X, y)
        visualizer = ClassPredictionError(model, classes=classes)
        visualizer.score(X, y)
        visualizer.finalize()

        # AppVeyor and Linux conda fail due to non-text-based differences
        # AppVeyor fails with RMS 13.161 - 13.289 (python - miniconda)
        self.assert_images_similar(visualizer, tol=12.5, windows_tol=13.3)

    def test_class_prediction_error_quickmethod(self):
        """
        Test the ClassPredictionError quickmethod
        """
        X, y = load_occupancy(return_dataset=True).to_numpy()

        fig = plt.figure()
        ax = fig.add_subplot()

        clf = SVC(random_state=42)
        viz = class_prediction_error(clf, X, y, ax=ax, show=False)

        # Not sure why the tolerance must be so high for this
        # Failing on travis with RMS 9.544
        # AppVeyor and Linux conda fail due to non-text-based differences: RMS 12.961
        # yellowbrick.exceptions.ImageComparisonFailure: images not close (RMS 15.538)
        self.assert_images_similar(viz, tol=16, windows_tol=16)

    def test_class_prediction_error_quickmethod_X_test_only(self):
        """
        Test the ClassPredictionError quickmethod
        """
        X, y = load_occupancy(return_dataset=True).to_numpy()
        X_train, X_test, y_train, y_test = tts(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )

        fig = plt.figure()
        ax = fig.add_subplot()

        clf = LinearSVC(random_state=42)
        with pytest.raises(
            YellowbrickValueError,
            match="must specify both X_test and y_test or neither",
        ):
            class_prediction_error(
                clf, X_train=X_train, y_train=y_train, X_test=X_test, ax=ax, show=False
            )

    def test_class_prediction_error_quickmethod_X_test_and_y_test(self):
        """
        Test the ClassPredictionError quickmethod
        """
        X, y = load_occupancy(return_dataset=True).to_numpy()
        X_train, X_test, y_train, y_test = tts(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )

        fig = plt.figure()
        ax = fig.add_subplot()

        clf = SVC(random_state=42)
        viz = class_prediction_error(
            clf,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            ax=ax,
            show=False,
        )

        # Not sure why the tolerance must be so high for this
        # Failing on travis with RMS 9.544
        # AppVeyor and Linux conda fail due to non-text-based differences: RMS 12.961
        self.assert_images_similar(viz, tol=13, windows_tol=13)

    @pytest.mark.filterwarnings("ignore:could not determine class_counts_")
    def test_classes_greater_than_indices(self):
        """
        A model error should be raised when there are more classes in fit than score
        """
        X, y = load_occupancy(return_dataset=True).to_numpy()
        classes = ["unoccupied", "occupied", "partytime"]

        model = LinearSVC(random_state=42)
        model.fit(X, y)
        with pytest.raises(ModelError):
            visualizer = ClassPredictionError(model, classes=classes)
            visualizer.score(X, y)

    def test_classes_less_than_indices(self):
        """
        Assert error when there is an attempt to filter classes
        """
        X, y = load_occupancy(return_dataset=True).to_numpy()
        classes = ["unoccupied"]

        model = LinearSVC(random_state=42)
        model.fit(X, y)
        with pytest.raises(NotImplementedError):
            visualizer = ClassPredictionError(model, classes=classes)
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
        with pytest.raises(YellowbrickValueError):
            visualizer = ClassPredictionError(model)
            visualizer.score(X, y)

    def test_score_returns_score(self):
        """
        Test that ClassPredictionError score() returns a score between 0 and 1
        """
        X, y = load_occupancy(return_dataset=True).to_numpy()

        # Create and fit the visualizer
        visualizer = ClassPredictionError(LinearSVC(random_state=42))
        visualizer.fit(X, y)

        # Score the visualizer
        s = visualizer.score(X, y)
        assert 0 <= s <= 1

    def test_with_fitted(self):
        """
        Test that visualizer properly handles an already-fitted model
        """
        X, y = load_occupancy(return_dataset=True).to_numpy()

        model = RandomForestClassifier().fit(X, y)
        classes = ["unoccupied", "occupied"]

        with patch.object(model, "fit") as mockfit:
            oz = ClassPredictionError(model, classes=classes)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with patch.object(model, "fit") as mockfit:
            oz = ClassPredictionError(model, classes=classes, is_fitted=True)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with patch.object(model, "fit") as mockfit:
            oz = ClassPredictionError(model, classes=classes, is_fitted=False)
            oz.fit(X, y)
            mockfit.assert_called_once_with(X, y)
