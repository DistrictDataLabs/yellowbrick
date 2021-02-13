# tests.test_model_selection.test_learning_curve
# Tests for the LearningCurve visualizer
#
# Author:   Jason Keung
# Created:  Tues May 23 11:45:00 2017 -0400
#
# Copyright (C) 2017 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_learning_curve.py jason.s.keung@gmail.com $

"""
Tests for the LearningCurve visualizer
"""

##########################################################################
# Imports
##########################################################################

import sys
import pytest
import numpy as np

from unittest.mock import patch
from tests.base import VisualTestCase

from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from yellowbrick.datasets import load_mushroom
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.model_selection.learning_curve import *

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
# LearningCurve Test Cases
##########################################################################


@pytest.mark.usefixtures("classification", "regression", "clusters")
class TestLearningCurve(VisualTestCase):
    """
    Test the LearningCurve visualizer
    """

    @patch.object(LearningCurve, "draw")
    def test_fit(self, mock_draw):
        """
        Assert that fit returns self and creates expected properties
        """
        X, y = self.classification
        params = (
            "train_sizes_",
            "train_scores_",
            "train_scores_mean_",
            "train_scores_std_",
            "test_scores_",
            "test_scores_mean_",
            "test_scores_std_",
        )

        oz = LearningCurve(GaussianNB(), random_state=12)
        for param in params:
            assert not hasattr(oz, param)

        assert oz.fit(X, y) is oz
        mock_draw.assert_called_once()

        for param in params:
            assert hasattr(oz, param)

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_classifier(self):
        """
        Test image closeness on a classification dataset
        """
        X, y = self.classification

        oz = LearningCurve(
            RandomForestClassifier(random_state=21), random_state=12
        ).fit(X, y)
        oz.finalize()

        self.assert_images_similar(oz, tol=0.1)  # w/o tol fails with RMS 0.037

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_regressor(self):
        """
        Test image closeness on a regression dataset
        """
        X, y = self.regression

        oz = LearningCurve(Ridge(), random_state=18)
        oz.fit(X, y)
        oz.finalize()

        self.assert_images_similar(oz)

    def test_clusters(self):
        """
        Test image closeness on a clustering dataset
        """
        X, y = self.clusters

        oz = LearningCurve(MiniBatchKMeans(random_state=281), random_state=182).fit(X)
        oz.finalize()

        self.assert_images_similar(oz, tol=10)

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_quick_method(self):
        """
        Test the learning curve quick method acts as expected
        """
        X, y = self.classification
        train_sizes = np.linspace(0.1, 1.0, 8)
        viz = learning_curve(
            GaussianNB(),
            X,
            y,
            train_sizes=train_sizes,
            cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=34),
            scoring="f1_macro",
            random_state=43,
            show=False
        )

        self.assert_images_similar(viz)

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_integration(self):
        """
        Test on a real dataset with pandas DataFrame and Series
        """
        data = load_mushroom(return_dataset=True)
        X, y = data.to_pandas()

        X = pd.get_dummies(X)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=32)
        oz = LearningCurve(GaussianNB(), cv=cv, random_state=23)
        oz.fit(X, y)
        oz.finalize()

        self.assert_images_similar(oz)

    def test_numpy_integration(self):
        """
        Test on a real dataset with NumPy arrays
        """
        data = load_mushroom(return_dataset=True)
        X, y = data.to_numpy()

        X = OneHotEncoder().fit_transform(X).toarray()

        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=32)
        oz = LearningCurve(GaussianNB(), cv=cv, random_state=23)
        oz.fit(X, y)
        oz.finalize()

        self.assert_images_similar(oz)

    @patch.object(LearningCurve, "draw")
    def test_reshape_scores(self, mock_draw):
        """
        Test supplying an alternate CV methodology and train_sizes
        """
        X, y = self.classification

        cv = ShuffleSplit(n_splits=12, test_size=0.2, random_state=14)
        oz = LearningCurve(LinearSVC(), cv=cv, train_sizes=[0.5, 0.75, 1.0])
        oz.fit(X, y)

        assert oz.train_scores_.shape == (3, 12)
        assert oz.test_scores_.shape == (3, 12)

    def test_bad_train_sizes(self):
        """
        Test learning curve with bad input for training size.
        """
        with pytest.raises(YellowbrickValueError):
            LearningCurve(LinearSVC(), train_sizes=10000)
