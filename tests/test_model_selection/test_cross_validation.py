# tests.test_model_selection.test_cross_validation
# Tests for the CVScores visualizer
#
# Author:  Rebecca Bilbro <bilbro@gmail.com>
# Created: Fri Aug 10 13:45:11 2018 -0400
#
# ID: test_cross_validation.py [] bilbro@gmail.com $

"""
Tests for the CVScores visualizer
"""

##########################################################################
## Imports
##########################################################################

import pytest

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin

from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit, StratifiedKFold

from yellowbrick.model_selection.cross_validation import *


try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


##########################################################################
## Test Cases
##########################################################################

@pytest.mark.usefixtures("classification", "regression")
class TestCrossValidation(VisualTestCase, DatasetMixin):
    """
    Test the CVScores visualizer
    """

    @patch.object(CVScores, 'draw')
    def test_fit(self, mock_draw):
        """
        Assert that fit returns self and creates expected properties
        """
        X, y = self.classification

        params = ("cv_scores_",  "cv_scores_mean_")

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

        oz = CVScores(
            KNeighborsClassifier(), cv=cv, scoring='f1_weighted',
        )

        oz.fit(X, y)
        oz.poof()

        self.assert_images_similar(oz)

    def test_regression(self):
        """
        Test image closeness on a regression dataset with a DecisionTree
        """
        X, y = self.regression

        cv = ShuffleSplit(3, random_state=938)

        oz = CVScores(
            DecisionTreeRegressor(random_state=23), cv=cv, scoring='r2',
        )

        oz.fit(X, y)
        oz.poof()

        self.assert_images_similar(oz)

    def test_quick_method(self):
        """
        Test validation curve quick method with image closeness on SVC
        """
        X, y = self.classification

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=321)
        ax = cv_scores(SVC(), X, y, cv=cv)

        self.assert_images_similar(ax=ax)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_integration(self):
        """
        Test on mushroom dataset with pandas DataFrame and Series and NB
        """
        df = self.load_pandas("mushroom")

        target = "target"
        features = [col for col in df.columns if col != target]

        X = pd.get_dummies(df[features])
        y = df[target]

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        cv = StratifiedKFold(n_splits=2, random_state=11)
        oz = CVScores(BernoulliNB(), cv=cv)

        oz.fit(X, y)
        oz.poof()

        self.assert_images_similar(oz)
        