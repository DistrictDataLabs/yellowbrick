# tests.test_classifier.test_rocauc
# Testing for the ROCAUC visualizer
#
# Author:   Benjamin Bengfort
# Author:   Rebecca Bilbro
# Created:  Tue May 23 13:41:55 2017 -0700
#
# Copyright (C) 2017 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_rocauc.py [] benjamin@bengfort.com $

"""
Testing for the ROCAUC visualizer
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np
import numpy.testing as npt

from unittest.mock import patch
from tests.base import VisualTestCase

from yellowbrick.classifier.rocauc import *
from yellowbrick.exceptions import ModelError
from yellowbrick.datasets import load_occupancy

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Fixtures
##########################################################################


class FakeClassifier(BaseEstimator, ClassifierMixin):
    """
    A fake classifier for testing noops on the visualizer.
    """

    pass


def assert_valid_rocauc_scores(visualizer, nscores=4):
    """
    Assertion helper to ensure scores are correctly computed
    """
    __tracebackhide__ = True
    assert len(visualizer.fpr.keys()) == nscores
    assert len(visualizer.tpr.keys()) == nscores
    assert len(visualizer.roc_auc.keys()) == nscores

    for k in (0, 1, "micro", "macro"):
        assert k in visualizer.fpr
        assert k in visualizer.tpr
        assert k in visualizer.roc_auc
        assert len(visualizer.fpr[k]) == len(visualizer.tpr[k])
        assert 0.0 < visualizer.roc_auc[k] < 1.0


##########################################################################
##  Tests
##########################################################################


@pytest.mark.usefixtures("binary", "multiclass")
class TestROCAUC(VisualTestCase):
    """
    Test ROCAUC visualizer
    """

    def test_binary_probability(self):
        """
        Test ROCAUC with a binary classifier with a predict_proba function
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(RandomForestClassifier(random_state=42))
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Score the visualizer
        s = visualizer.score(self.binary.X.test, self.binary.y.test)

        # Test that score method successfully returns a value between 0 and 1
        assert 0 <= s <= 1

        # Check the scores
        assert_valid_rocauc_scores(visualizer)

        # Compare the images
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_binary_probability_decision(self):
        """
        Test ROCAUC with a binary classifier with both decision & predict_proba
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(AdaBoostClassifier())
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Score the visualizer
        s = visualizer.score(self.binary.X.test, self.binary.y.test)

        # Test that score method successfully returns a value between 0 and 1
        assert 0 <= s <= 1

        # Check the scores
        assert_valid_rocauc_scores(visualizer)

        # Compare the images
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_binary_probability_decision_single_curve(self):
        """
        Test ROCAUC binary classifier with both decision & predict_proba with per_class=False
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(
            AdaBoostClassifier(), micro=False, macro=False, per_class=False
        )
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Score the visualizer
        s = visualizer.score(self.binary.X.test, self.binary.y.test)

        # Test that score method successfully returns a value between 0 and 1
        assert 0 <= s <= 1

        # Check the scores
        assert len(visualizer.fpr.keys()) == 1
        assert len(visualizer.tpr.keys()) == 1
        assert len(visualizer.roc_auc.keys()) == 1

        # Compare the images
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_binary_decision(self):
        """
        Test ROCAUC with a binary classifier with a decision_function
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(
            LinearSVC(random_state=42), micro=False, macro=False, per_class=False
        )
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Score the visualizer
        s = visualizer.score(self.binary.X.test, self.binary.y.test)

        # Test that score method successfully returns a value between 0 and 1
        assert 0 <= s <= 1

        # Check the scores
        assert len(visualizer.fpr.keys()) == 1
        assert len(visualizer.tpr.keys()) == 1
        assert len(visualizer.roc_auc.keys()) == 1

        # Compare the images
        # NOTE: increased tolerance for both AppVeyor and Travis CI tests
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=10)

    def test_binary_decision_per_class(self):
        """
        Test ROCAUC with a binary classifier with a decision_function
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(
            LinearSVC(random_state=42), micro=False, macro=False, per_class=True
        )
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Score the visualizer
        s = visualizer.score(self.binary.X.test, self.binary.y.test)

        # Test that score method successfully returns a value between 0 and 1
        assert 0 <= s <= 1

        # Check the scores
        assert len(visualizer.fpr.keys()) == 2
        assert len(visualizer.tpr.keys()) == 2
        assert len(visualizer.roc_auc.keys()) == 2

        # Compare the images
        # NOTE: increased tolerance for both AppVeyor and Travis CI tests
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=10)

    def test_binary_micro_error(self):
        """
        Test ROCAUC to see if _binary_decision with micro = True raises an error
        """
        # Create visualizer with a linear model to force a binary decision
        visualizer = ROCAUC(LinearSVC(random_state=42), micro=True, per_class=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Ensure score raises error (micro curves aren't defined for binary decisions)
        with pytest.raises(ModelError):
            visualizer.score(self.binary.X.test, self.binary.y.test)

    def test_binary_macro_error(self):
        """
        Test ROCAUC to see if _binary_decision with macro = True raises an error
        """
        # Create visualizer with a linear model to force a binary decision
        visualizer = ROCAUC(LinearSVC(random_state=42), macro=True, per_class=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Ensure score raises error (macro curves aren't defined for binary decisions)
        with pytest.raises(ModelError):
            visualizer.score(self.binary.X.test, self.binary.y.test)

    def test_multiclass_rocauc(self):
        """
        Test ROCAUC with a multiclass classifier
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(GaussianNB())
        visualizer.fit(self.multiclass.X.train, self.multiclass.y.train)

        # Score the visualizer
        s = visualizer.score(self.multiclass.X.test, self.multiclass.y.test)

        # Test that score method successfully returns a value between 0 and 1
        assert 0 <= s <= 1

        # Check the scores
        assert_valid_rocauc_scores(visualizer, nscores=8)

        # Compare the images
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_rocauc_no_classes(self):
        """
        Test ROCAUC without per-class curves
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(GaussianNB(), per_class=False)
        visualizer.fit(self.multiclass.X.train, self.multiclass.y.train)

        # Score the visualizer (should be the micro average)
        s = visualizer.score(self.multiclass.X.test, self.multiclass.y.test)
        assert s == pytest.approx(0.77303, abs=1e-4)

        # Assert that there still are per-class scores
        for c in (0, 1):
            assert c in visualizer.fpr
            assert c in visualizer.tpr
            assert c in visualizer.roc_auc

        # Compare the images
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_rocauc_no_curves(self):
        """
        Test ROCAUC with no curves specified at all
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(GaussianNB(), per_class=False, macro=False, micro=False)
        visualizer.fit(self.multiclass.X.train, self.multiclass.y.train)

        # Attempt to score the visualizer
        with pytest.raises(YellowbrickValueError, match="no curves will be drawn"):
            visualizer.score(self.multiclass.X.test, self.multiclass.y.test)

    def test_rocauc_quickmethod(self):
        """
        Test the ROCAUC quick method
        """
        X, y = load_occupancy(return_dataset=True).to_numpy()
        model = LogisticRegression()

        # compare the images
        visualizer = roc_auc(model, X, y, show=False)
        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_integration(self):
        """
        Test the ROCAUC with Pandas dataframe
        """
        X, y = load_occupancy(return_dataset=True).to_pandas()

        # Create train/test splits
        splits = tts(X, y, test_size=0.2, random_state=4512)
        X_train, X_test, y_train, y_test = splits

        visualizer = ROCAUC(GaussianNB())
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)

        # Compare the images
        visualizer.finalize()
        self.assert_images_similar(visualizer)

    def test_rocauc_no_micro(self):
        """
        Test ROCAUC without a micro average
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression(), micro=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Score the visualizer (should be the macro average)
        s = visualizer.score(self.binary.X.test, self.binary.y.test)
        assert s == pytest.approx(0.8661, abs=1e-4)

        # Assert that there is no micro score
        assert "micro" not in visualizer.fpr
        assert "micro" not in visualizer.tpr
        assert "micro" not in visualizer.roc_auc

        # Compare the images
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_rocauc_no_macro(self):
        """
        Test ROCAUC without a macro average
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression(), macro=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Score the visualizer (should be the micro average)
        s = visualizer.score(self.binary.X.test, self.binary.y.test)
        assert s == pytest.approx(0.8573, abs=1e-4)

        # Assert that there is no macro score
        assert "macro" not in visualizer.fpr
        assert "macro" not in visualizer.tpr
        assert "macro" not in visualizer.roc_auc

        # Compare the images
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_rocauc_no_macro_no_micro(self):
        """
        Test ROCAUC without a macro or micro average
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression(), macro=False, micro=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Score the visualizer (should be the F1 score)
        s = visualizer.score(self.binary.X.test, self.binary.y.test)
        assert s == pytest.approx(0.8)

        # Assert that there is no macro score
        assert "macro" not in visualizer.fpr
        assert "macro" not in visualizer.tpr
        assert "macro" not in visualizer.roc_auc

        # Assert that there is no micro score
        assert "micro" not in visualizer.fpr
        assert "micro" not in visualizer.tpr
        assert "micro" not in visualizer.roc_auc

        # Compare the images
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.1, windows_tol=10)

    def test_rocauc_label_encoded(self):
        """
        Test ROCAUC with a target specifying a list of classes as strings
        """
        class_labels = ["a", "b", "c", "d", "e", "f"]

        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression(), classes=class_labels)
        visualizer.fit(self.multiclass.X.train, self.multiclass.y.train)

        # Score the visualizer
        visualizer.score(self.multiclass.X.test, self.multiclass.y.test)
        assert list(visualizer.classes_) == class_labels

    def test_rocauc_not_label_encoded(self):
        """
        Test ROCAUC with a target whose classes are unencoded strings before scoring
        """
        # Map numeric targets to strings
        classes = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f"}
        y_train = np.array([classes[yi] for yi in self.multiclass.y.train])
        y_test = np.array([classes[yi] for yi in self.multiclass.y.test])

        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression())
        visualizer.fit(self.multiclass.X.train, y_train)

        # Confirm that y_train and y_test have the same targets before calling score
        assert set(y_train) == set(y_test)

    def test_binary_decision_function_rocauc(self):
        """
        Test ROCAUC with binary classifiers that have a decision function
        """
        # Load the model and assert there is no predict_proba method.
        model = LinearSVC()
        with pytest.raises(AttributeError):
            model.predict_proba

        # Fit model and visualizer
        visualizer = ROCAUC(model)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # First 10 expected values in the y_scores
        first_ten_expected = np.asarray(
            [-0.092, 0.019, -0.751, -0.838, 0.183, -0.344, -1.019, 2.203, 1.415, -0.529]
        )

        # Get the predict_proba scores and evaluate
        y_scores = visualizer._get_y_scores(self.binary.X.train)

        # Check to see if the first 10 y_scores match the expected
        npt.assert_array_almost_equal(y_scores[:10], first_ten_expected, decimal=1)

    def test_binary_false_decision_function_error(self):
        """
        Test binary decision_function model raises error when the binary param is False
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(LinearSVC(random_state=42), binary=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Ensure score raises error
        # (only binary curve defined for binary decisions with decision_function clf)
        with pytest.raises(ModelError):
            visualizer.score(self.binary.X.test, self.binary.y.test)

    def test_multi_decision_function_rocauc(self):
        """
        Test ROCAUC with multiclass classifiers that have a decision function
        """
        # Load the model and assert there is no predict_proba method.
        model = LinearSVC()
        with pytest.raises(AttributeError):
            model.predict_proba

        # Fit model and visualizer
        visualizer = ROCAUC(model)
        visualizer.fit(self.multiclass.X.train, self.multiclass.y.train)

        # First 5 expected arrays in the y_scores
        first_five_expected = [
            [-0.370, -0.543, -1.059, -0.466, -0.743, -1.156],
            [-0.445, -0.693, -0.362, -1.002, -0.815, -0.878],
            [-1.058, -0.808, -0.291, -0.767, -0.651, -0.586],
            [-0.446, -1.255, -0.489, -0.961, -0.807, -0.126],
            [-1.066, -0.493, -0.639, -0.442, -0.639, -1.017],
        ]

        # Get the predict_proba scores and evaluate
        y_scores = visualizer._get_y_scores(self.multiclass.X.train)

        # Check to see if the first 5 y_score arrays match the expected
        npt.assert_array_almost_equal(y_scores[:5], first_five_expected, decimal=1)

    def test_predict_proba_rocauc(self):
        """
        Test ROCAUC with classifiers that utilize predict_proba
        """
        # Load the model and assert there is no decision_function method.
        model = GaussianNB()
        with pytest.raises(AttributeError):
            model.decision_function

        # Fit model and visualizer
        visualizer = ROCAUC(model)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # First 10 expected arrays in the y_scores
        first_ten_expected = np.asarray(
            [
                [0.595, 0.405],
                [0.161, 0.839],
                [0.990, 0.010],
                [0.833, 0.167],
                [0.766, 0.234],
                [0.996, 0.004],
                [0.592, 0.408],
                [0.007, 0.993],
                [0.035, 0.965],
                [0.764, 0.236],
            ]
        )

        # Get the predict_proba scores and evaluate
        y_scores = visualizer._get_y_scores(self.binary.X.train)

        # Check to see if the first 10 y_score arrays match the expected
        npt.assert_array_almost_equal(y_scores[:10], first_ten_expected, decimal=1)

    def test_no_scoring_function(self):
        """
        Test ROCAUC with classifiers that have no scoring method
        """
        visualizer = ROCAUC(FakeClassifier())
        with pytest.raises(ModelError):
            visualizer._get_y_scores(self.binary.X.train)

    def test_with_fitted(self):
        """
        Test that visualizer properly handles an already-fitted model
        """
        X, y = load_occupancy(return_dataset=True).to_numpy()

        model = GaussianNB().fit(X, y)
        classes = ["unoccupied", "occupied"]

        with patch.object(model, "fit") as mockfit:
            oz = ROCAUC(model, classes=classes)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with patch.object(model, "fit") as mockfit:
            oz = ROCAUC(model, classes=classes, is_fitted=True)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with patch.object(model, "fit") as mockfit:
            oz = ROCAUC(model, classes=classes, is_fitted=False)
            oz.fit(X, y)
            mockfit.assert_called_once_with(X, y)

    def test_binary_meta_param(self):
        """
        Test the binary meta param with ROCAUC
        """
        oz = ROCAUC(GaussianNB(), binary=False)
        assert oz.micro is True
        assert oz.macro is True
        assert oz.per_class is True

        oz = ROCAUC(GaussianNB(), binary=True)
        assert oz.micro is False
        assert oz.macro is False
        assert oz.per_class is False
