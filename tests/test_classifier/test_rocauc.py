# tests.test_classifier.test_rocauc
# Testing for the ROCAUC visualizer
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  Tue May 23 13:41:55 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_rocauc.py [] benjamin@bengfort.com $

"""
Testing for the ROCAUC visualizer
"""

##########################################################################
## Imports
##########################################################################

import os
import pytest
import numpy as np
import numpy.testing as npt

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin
from yellowbrick.classifier.rocauc import *
from yellowbrick.exceptions import ModelError, YellowbrickValueError

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


##########################################################################
## Fixtures
##########################################################################

# Increased tolerance for AppVeyor tests
TOL = 10 if os.name == 'nt'  else 0.1


class FakeClassifier(BaseEstimator, ClassifierMixin):
    """
    A fake classifier for testing noops on the visualizer.
    """
    pass


##########################################################################
##  Tests
##########################################################################

@pytest.mark.usefixtures("binary", "multiclass")
class ROCAUCTests(VisualTestCase, DatasetMixin):

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
        self.assertEqual(len(visualizer.fpr.keys()), 4)
        self.assertEqual(len(visualizer.tpr.keys()), 4)
        self.assertEqual(len(visualizer.roc_auc.keys()), 4)

        for k in (0, 1, "micro", "macro"):
            self.assertIn(k, visualizer.fpr)
            self.assertIn(k, visualizer.tpr)
            self.assertIn(k, visualizer.roc_auc)
            self.assertEqual(len(visualizer.fpr[k]), len(visualizer.tpr[k]))
            self.assertGreater(visualizer.roc_auc[k], 0.0)
            self.assertLess(visualizer.roc_auc[k], 1.0)

        # Compare the images
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=TOL)

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
        self.assertEqual(len(visualizer.fpr.keys()), 4)
        self.assertEqual(len(visualizer.tpr.keys()), 4)
        self.assertEqual(len(visualizer.roc_auc.keys()), 4)

        for k in (0, 1, "micro", "macro"):
            self.assertIn(k, visualizer.fpr)
            self.assertIn(k, visualizer.tpr)
            self.assertIn(k, visualizer.roc_auc)
            self.assertEqual(len(visualizer.fpr[k]), len(visualizer.tpr[k]))
            self.assertGreater(visualizer.roc_auc[k], 0.0)
            self.assertLess(visualizer.roc_auc[k], 1.0)

        # Compare the images
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=TOL)

    def test_binary_decision(self):
        """
        Test ROCAUC with a binary classifier with a decision_function
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(LinearSVC(random_state=42), micro=False, macro=False, per_class=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Score the visualizer
        s = visualizer.score(self.binary.X.test, self.binary.y.test)

        # Test that score method successfully returns a value between 0 and 1
        assert 0 <= s <= 1

        # Check the scores
        self.assertEqual(len(visualizer.fpr.keys()), 1)
        self.assertEqual(len(visualizer.tpr.keys()), 1)
        self.assertEqual(len(visualizer.roc_auc.keys()), 1)

        # Compare the images
        # NOTE: increased tolerance for both AppVeyor and Travis CI tests
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=10)

    def test_binary_micro_error(self):
        """
        Test ROCAUC to see if _binary_decision with micro = True raises an error
        """
        # Create visualizer with a linear model to force a binary decision
        visualizer = ROCAUC(LinearSVC(random_state=42), micro=True)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Ensure score raises error (micro curves aren't defined for binary decisions)
        with self.assertRaises(ModelError):
            visualizer.score(self.binary.X.test, self.binary.y.test)

    def test_binary_macro_error(self):
        """
        Test ROCAUC to see if _binary_decision with macro = True raises an error
        """
        # Create visualizer with a linear model to force a binary decision
        visualizer = ROCAUC(LinearSVC(random_state=42), macro=True)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Ensure score raises error (macro curves aren't defined for binary decisions)
        with self.assertRaises(ModelError):
            visualizer.score(self.binary.X.test, self.binary.y.test)

    def test_binary_per_class_error(self):
        """
        Test ROCAUC to see if _binary_decision with per_class = True raises an error
        """
        # Create visualizer with a linear model to force a binary decision
        visualizer = ROCAUC(LinearSVC(random_state=42), per_class=True)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Ensure score raises error (per_class curves not defined for binary decisions)
        with self.assertRaises(ModelError):
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
        self.assertEqual(len(visualizer.fpr.keys()), 8)
        self.assertEqual(len(visualizer.tpr.keys()), 8)
        self.assertEqual(len(visualizer.roc_auc.keys()), 8)

        for k in (0, 1, "micro", "macro"):
            self.assertIn(k, visualizer.fpr)
            self.assertIn(k, visualizer.tpr)
            self.assertIn(k, visualizer.roc_auc)
            self.assertEqual(len(visualizer.fpr[k]), len(visualizer.tpr[k]))
            self.assertGreater(visualizer.roc_auc[k], 0.0)
            self.assertLess(visualizer.roc_auc[k], 1.0)

        # Compare the images
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=TOL)

    def test_rocauc_quickmethod(self):
        """
        Test the ROCAUC quick method
        """
        data = load_breast_cancer()
        model = DecisionTreeClassifier()

        # TODO: image comparison of the quick method
        roc_auc(model, data.data, data.target)

    def test_rocauc_no_micro(self):
        """
        Test ROCAUC without a micro average
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression(), micro=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Score the visualizer (should be the macro average)
        s = visualizer.score(self.binary.X.test, self.binary.y.test)
        self.assertAlmostEqual(s, 0.8)

        # Assert that there is no micro score
        self.assertNotIn("micro", visualizer.fpr)
        self.assertNotIn("micro", visualizer.tpr)
        self.assertNotIn("micro", visualizer.roc_auc)

        # Compare the images
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=TOL)

    def test_rocauc_no_macro(self):
        """
        Test ROCAUC without a macro average
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression(), macro=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Score the visualizer (should be the micro average)
        s = visualizer.score(self.binary.X.test, self.binary.y.test)
        self.assertAlmostEqual(s, 0.8)

        # Assert that there is no macro score
        self.assertNotIn("macro", visualizer.fpr)
        self.assertNotIn("macro", visualizer.tpr)
        self.assertNotIn("macro", visualizer.roc_auc)

        # Compare the images
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=TOL)

    def test_rocauc_no_macro_no_micro(self):
        """
        Test ROCAUC without a macro or micro average
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression(), macro=False, micro=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Score the visualizer (should be the F1 score)
        s = visualizer.score(self.binary.X.test, self.binary.y.test)
        self.assertAlmostEqual(s, 0.8)

        # Assert that there is no macro score
        self.assertNotIn("macro", visualizer.fpr)
        self.assertNotIn("macro", visualizer.tpr)
        self.assertNotIn("macro", visualizer.roc_auc)

        # Assert that there is no micro score
        self.assertNotIn("micro", visualizer.fpr)
        self.assertNotIn("micro", visualizer.tpr)
        self.assertNotIn("micro", visualizer.roc_auc)

        # Compare the images
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=TOL)

    def test_rocauc_no_classes(self):
        """
        Test ROCAUC without per-class curves
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression(), per_class=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Score the visualizer (should be the micro average)
        s = visualizer.score(self.binary.X.test, self.binary.y.test)
        self.assertAlmostEqual(s, 0.8)

        # Assert that there still are per-class scores
        for c in (0, 1):
            self.assertIn(c, visualizer.fpr)
            self.assertIn(c, visualizer.tpr)
            self.assertIn(c, visualizer.roc_auc)

        # Compare the images
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=TOL)

    def test_rocauc_no_curves(self):
        """
        Test ROCAUC with no curves specified at all
        """
        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression(), per_class=False, macro=False, micro=False)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # Attempt to score the visualizer
        with pytest.raises(YellowbrickValueError, match="no curves will be drawn"):
            visualizer.score(self.binary.X.test, self.binary.y.test)

    def test_rocauc_label_encoded(self):
        """
        Test ROCAUC with a target specifying a list of classes as strings
        """
        class_labels = ['a', 'b', 'c', 'd', 'e', 'f']

        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression(), classes=class_labels)
        visualizer.fit(self.multiclass.X.train, self.multiclass.y.train)

        # Score the visualizer
        visualizer.score(self.multiclass.X.test, self.multiclass.y.test)
        self.assertEqual(list(visualizer.classes_), class_labels)

    def test_rocauc_not_label_encoded(self):
        """
        Test ROCAUC with a target whose classes are unencoded strings before scoring
        """
        # Map numeric targets to strings
        classes = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f'}
        y_train = np.array([classes[yi] for yi in self.multiclass.y.train])
        y_test = np.array([classes[yi] for yi in self.multiclass.y.test])

        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression())
        visualizer.fit(self.multiclass.X.train, y_train)

        # Confirm that y_train and y_test have the same targets before calling score
        self.assertEqual(set(y_train), set(y_test))

    def test_binary_decision_function_rocauc(self):
        """
        Test ROCAUC with binary classifiers that have a decision function
        """
        # Load the model and assert there is no predict_proba method.
        model = LinearSVC()
        with self.assertRaises(AttributeError):
            model.predict_proba

        # Fit model and visualizer
        visualizer = ROCAUC(model)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # First 10 expected values in the y_scores
        first_ten_expected = np.asarray([
            -0.092, 0.019, -0.751, -0.838, 0.183, -0.344, -1.019, 2.203, 1.415, -0.529
        ])

        # Get the predict_proba scores and evaluate
        y_scores = visualizer._get_y_scores(self.binary.X.train)

        # Check to see if the first 10 y_scores match the expected
        npt.assert_array_almost_equal(y_scores[:10], first_ten_expected, decimal=1)

    def test_multi_decision_function_rocauc(self):
        """
        Test ROCAUC with multiclass classifiers that have a decision function
        """
        # Load the model and assert there is no predict_proba method.
        model = LinearSVC()
        with self.assertRaises(AttributeError):
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
             [-1.066, -0.493, -0.639, -0.442, -0.639, -1.017]
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
        with self.assertRaises(AttributeError):
            model.decision_function

        # Fit model and visualizer
        visualizer = ROCAUC(model)
        visualizer.fit(self.binary.X.train, self.binary.y.train)

        # First 10 expected arrays in the y_scores
        first_ten_expected = np.asarray([
            [0.595, 0.405],
            [0.161, 0.839],
            [0.990, 0.010],
            [0.833, 0.167],
            [0.766, 0.234],
            [0.996, 0.004],
            [0.592, 0.408],
            [0.007, 0.993],
            [0.035, 0.965],
            [0.764, 0.236]
        ])

        # Get the predict_proba scores and evaluate
        y_scores = visualizer._get_y_scores(self.binary.X.train)

        # Check to see if the first 10 y_score arrays match the expected
        npt.assert_array_almost_equal(y_scores[:10], first_ten_expected, decimal=1)

    def test_no_scoring_function(self):
        """
        Test ROCAUC with classifiers that have no scoring method
        """
        visualizer = ROCAUC(FakeClassifier())
        with self.assertRaises(ModelError):
            visualizer._get_y_scores(self.binary.X.train)
