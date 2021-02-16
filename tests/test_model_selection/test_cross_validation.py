# tests.test_model_selection.test_cross_validation
# Tests for the CVScores visualizer
#
# Author:  Rebecca Bilbro
# Created: Fri Aug 10 13:45:11 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_cross_validation.py [962c8bb] rebeccabilbro@users.noreply.github.com $

"""
Tests for the CVScores visualizer
"""

##########################################################################
# Imports
##########################################################################

import pytest
import numpy.testing as npt

from unittest.mock import patch
from tests.base import VisualTestCase

from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import ShuffleSplit, StratifiedKFold

from yellowbrick.datasets import load_mushroom
from yellowbrick.model_selection.cross_validation import *


try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
# Test Cases
##########################################################################


@pytest.mark.usefixtures("classification", "regression")
class TestCrossValidation(VisualTestCase):
    """
    Test the CVScores visualizer
    """

    @patch.object(CVScores, "draw")
    def test_fit(self, mock_draw):
        """
        Assert that fit returns self and creates expected properties
        """
        X, y = self.classification

        params = ("cv_scores_", "cv_scores_mean_")

        oz = CVScores(SVC())

        for param in params:
            assert not hasattr(oz, param)

        assert oz.fit(X, y) is oz
        mock_draw.assert_called_once()

        for param in params:
            assert hasattr(oz, param)

    def test_classifier(self):
        """
        Test image closeness on a classification dataset with kNN
        """
        X, y = self.classification

        cv = ShuffleSplit(3, random_state=288)

        oz = CVScores(KNeighborsClassifier(), cv=cv, scoring="f1_weighted")

        oz.fit(X, y)
        oz.finalize()

        self.assert_images_similar(oz, tol=2.0)

    def test_classifier_with_cv(self):
        """
        Test that CVScores passes cv param to internal sklearn classifier with CV
        """
        X, y = self.classification

        cv = StratifiedKFold(5, shuffle=True, random_state=288)
        oz_external_cv = CVScores(LogisticRegressionCV(), cv=cv)
        oz_internal_cv = CVScores(LogisticRegressionCV(cv=cv))
        oz_external_cv.fit(X, y)
        oz_internal_cv.fit(X, y)

        npt.assert_array_almost_equal(
            oz_external_cv.cv_scores_, oz_internal_cv.cv_scores_, decimal=1
        )

    def test_regression(self):
        """
        Test image closeness on a regression dataset with a DecisionTree
        """
        X, y = self.regression

        cv = ShuffleSplit(5, random_state=938)
        oz = CVScores(DecisionTreeRegressor(random_state=23), cv=cv, scoring="r2")
        oz.fit(X, y)
        oz.finalize()

        self.assert_images_similar(oz, tol=36.0)

    def test_regressor_with_cv(self):
        """
        Test that CVScores passes cv param to internal sklearn regressor with CV
        """
        X, y = self.regression

        cv = ShuffleSplit(5, random_state=288)
        oz_external_cv = CVScores(RidgeCV(), cv=cv)
        oz_internal_cv = CVScores(RidgeCV(cv=cv))
        oz_external_cv.fit(X, y)
        oz_internal_cv.fit(X, y)

        npt.assert_array_almost_equal(
            oz_external_cv.cv_scores_, oz_internal_cv.cv_scores_
        )

    def test_quick_method(self):
        """
        Test cross validation quick method with image closeness on SVC
        """
        X, y = self.classification

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=321)
        viz = cv_scores(SVC(), X, y, cv=cv, show=False)

        self.assert_images_similar(viz, tol=2.0)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_integration(self):
        """
        Test on mushroom dataset with pandas DataFrame and Series and NB
        """
        data = load_mushroom(return_dataset=True)
        X, y = data.to_pandas()

        X = pd.get_dummies(X)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=11)
        oz = CVScores(BernoulliNB(), cv=cv)

        oz.fit(X, y)
        oz.finalize()

        self.assert_images_similar(oz, tol=2.0)

    def test_numpy_integration(self):
        """
        Test on mushroom dataset with NumPy arrays
        """
        data = load_mushroom(return_dataset=True)
        X, y = data.to_numpy()

        X = OneHotEncoder().fit_transform(X).toarray()

        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=11)
        oz = CVScores(BernoulliNB(), cv=cv)

        oz.fit(X, y)
        oz.finalize()

        self.assert_images_similar(oz, tol=2.0)
