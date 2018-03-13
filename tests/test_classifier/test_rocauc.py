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

import pytest
import numpy as np
import numpy.testing as npt

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin
from yellowbrick.classifier.rocauc import *
from yellowbrick.exceptions import ModelError

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split as tts


##########################################################################
## Data
##########################################################################

X = np.array(
        [[ 2.318, 2.727, 4.260, 7.212, 4.792],
         [ 2.315, 2.726, 4.295, 7.140, 4.783,],
         [ 2.315, 2.724, 4.260, 7.135, 4.779,],
         [ 2.110, 3.609, 4.330, 7.985, 5.595,],
         [ 2.110, 3.626, 4.330, 8.203, 5.621,],
         [ 2.110, 3.620, 4.470, 8.210, 5.612,]]
    )

yb = np.array([1, 1, 0, 1, 0, 0])
ym = np.array([1, 0, 2, 1, 2, 0])


##########################################################################
## Fixtures
##########################################################################

class FakeClassifier(BaseEstimator, ClassifierMixin):
    """
    A fake classifier for testing noops on the visualizer.
    """
    pass


##########################################################################
##  Tests
##########################################################################

class ROCAUCTests(VisualTestCase, DatasetMixin):

    def load_binary_data(self):
        """
        Returns the binary test data set.
        """
        # Load the Data
        data = self.load_data("occupancy")

        X = data[[
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]]

        y = data['occupancy'].astype(int)

        # Convert X to an ndarray
        X = X.copy().view((float, len(X.dtype.names)))

        # Return train/test splits
        return tts(X, y, test_size=0.2, random_state=42)

    def load_multiclass_data(self):
        """
        Returns the multiclass test data set.
        """
        raise NotImplementedError("Need to add multiclass data soon!")

    @pytest.mark.skip(reason="binary classifiers don't currently work as expected")
    def test_binary_rocauc(self):
        """
        Test ROCAUC with a binary classifier
        """
        X_train, X_test, y_train, y_test = self.load_binary_data()

        # Create and fit the visualizer
        visualizer = ROCAUC(LinearSVC())
        visualizer.fit(X_train, y_train)

        # Score the visualizer
        s = visualizer.score(X_test, y_test)
        self.assertAlmostEqual(s, 0.93230159261495249)

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
        self.assert_images_similar(visualizer)

    @pytest.mark.xfail(reason="see issue #315")
    def test_multiclass_rocauc(self):
        """
        Test ROCAUC with a multiclass classifier
        """
        # Load the Data
        # TODO: Switch to a true multi-class dataset
        X_train, X_test, y_train, y_test = self.load_binary_data()

        # Create and fit the visualizer
        visualizer = ROCAUC(MultinomialNB())
        visualizer.fit(X_train, y_train)

        # Score the visualizer
        s = visualizer.score(X_test, y_test)
        self.assertAlmostEqual(s, 0.93230159261495249)

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
        self.assert_images_similar(visualizer, tol=0.071)

    def test_rocauc_quickmethod(self):
        """
        Test the ROCAUC quick method
        """
        data = load_breast_cancer()
        model = DecisionTreeClassifier()

        # TODO: impage comparison of the quick method
        roc_auc(model, data.data, data.target)

    @pytest.mark.xfail(reason="see issue #315")
    def test_rocauc_no_micro(self):
        """
        Test ROCAUC without a micro average
        """
        # Load the Data
        X_train, X_test, y_train, y_test = self.load_binary_data()

        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression(), micro=False)
        visualizer.fit(X_train, y_train)

        # Score the visualizer (should be the macro average)
        s = visualizer.score(X_test, y_test)
        self.assertAlmostEqual(s, 0.99578564759755916)

        # Assert that there is no micro score
        self.assertNotIn("micro", visualizer.fpr)
        self.assertNotIn("micro", visualizer.tpr)
        self.assertNotIn("micro", visualizer.roc_auc)

        # Compare the images
        visualizer.poof()
        self.assert_images_similar(visualizer)

    @pytest.mark.xfail(reason="see issue #315")
    def test_rocauc_no_macro(self):
        """
        Test ROCAUC without a macro average
        """
        # Load the Data
        X_train, X_test, y_train, y_test = self.load_binary_data()

        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression(), macro=False)
        visualizer.fit(X_train, y_train)

        # Score the visualizer (should be the micro average)
        s = visualizer.score(X_test, y_test)
        self.assertAlmostEqual(s, 0.99766508576965574)

        # Assert that there is no macro score
        self.assertNotIn("macro", visualizer.fpr)
        self.assertNotIn("macro", visualizer.tpr)
        self.assertNotIn("macro", visualizer.roc_auc)

        # Compare the images
        visualizer.poof()
        self.assert_images_similar(visualizer)

    def test_rocauc_no_macro_no_micro(self):
        """
        Test ROCAUC without a macro or micro average
        """
        # Load the Data
        X_train, X_test, y_train, y_test = self.load_binary_data()

        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression(), macro=False, micro=False)
        visualizer.fit(X_train, y_train)

        # Score the visualizer (should be the F1 score)
        s = visualizer.score(X_test, y_test)
        self.assertAlmostEqual(s, 0.98978599221789887)

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
        self.assert_images_similar(visualizer)

    @pytest.mark.xfail(reason="see issue #315")
    def test_rocauc_no_classes(self):
        """
        Test ROCAUC without per-class curves
        """
        # Load the Data
        X_train, X_test, y_train, y_test = self.load_binary_data()

        # Create and fit the visualizer
        visualizer = ROCAUC(LogisticRegression(), per_class=False)
        visualizer.fit(X_train, y_train)

        # Score the visualizer (should be the micro average)
        s = visualizer.score(X_test, y_test)
        self.assertAlmostEqual(s, 0.99766508576965574)

        # Assert that there still are per-class scores
        for c in (0, 1):
            self.assertIn(c, visualizer.fpr)
            self.assertIn(c, visualizer.tpr)
            self.assertIn(c, visualizer.roc_auc)

        # Compare the images
        visualizer.poof()
        self.assert_images_similar(visualizer)

    def test_rocauc_no_curves(self):
        """
        Test ROCAUC with no curves specified at all
        """
        # Load the Data
        X_train, X_test, y_train, y_test = self.load_binary_data()

        # Create and fit the visualizer
        visualizer = ROCAUC(
            LogisticRegression(), per_class=False, macro=False, micro=False
        )
        visualizer.fit(X_train, y_train)

        # TODO: Raise an exception in this case.

        # Compare the images - should be blank
        visualizer.poof()
        self.assert_images_similar(visualizer)

    @pytest.mark.skip(reason="not implemented yet")
    def test_rocauc_label_encoded(self):
        """
        Test ROCAUC with label encoding before scoring
        """
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_rocauc_not_label_encoded(self):
        """
        Test ROCAUC without label encoding before scoring
        """
        pass

    @pytest.mark.xfail(reason="not working with expected precision")
    def test_decision_function_rocauc(self):
        """
        Test ROCAUC with classifiers that have a decision function
        """
        # Load the model and assert there is no predict_proba method.
        model = LinearSVC()
        with self.assertRaises(AttributeError):
            model.predict_proba

        # Fit model and visualizer
        visualizer = ROCAUC(model)
        visualizer.fit(X, yb)

        expected = np.asarray([
            0.204348,  0.228593,  0.219908, -0.211756, -0.26155 , -0.221405
        ])

        # Get the predict_proba scores and evaluate
        y_scores = visualizer._get_y_scores(X)
        npt.assert_array_almost_equal(y_scores, expected, decimal=1)

    def test_predict_proba_rocauc(self):
        """
        Test ROCAUC with classifiers that utilize predict_proba
        """
        # Load the model and assert there is no decision_function method.
        model = MultinomialNB()
        with self.assertRaises(AttributeError):
            model.decision_function

        # Fit model and visualizer
        visualizer = ROCAUC(model)
        visualizer.fit(X, yb)

        expected = np.asarray([
            [0.493788,  0.506212],
            [0.493375,  0.506625],
            [0.493572,  0.506428],
            [0.511063,  0.488936],
            [0.511887,  0.488112],
            [0.510841,  0.489158],
        ])

        # Get the predict_proba scores and evaluate
        y_scores = visualizer._get_y_scores(X)
        npt.assert_array_almost_equal(y_scores, expected)

    def test_no_scoring_function(self):
        """
        Test ROCAUC with classifiers that have no scoring method
        """
        visualizer = ROCAUC(FakeClassifier())
        with self.assertRaises(ModelError):
            visualizer._get_y_scores(X)
