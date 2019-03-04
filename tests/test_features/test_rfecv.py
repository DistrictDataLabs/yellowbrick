# tests.test_feautures.test_rfecv
# Tests for the RFECV visualizer
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Tue Apr 03 17:35:16 2018 -0400
#
# ID: test_rfecv.py [] benjamin@bengfort.com $

"""
Tests for the RFECV visualizer
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest
import numpy as np
import numpy.testing as npt

from unittest.mock import patch
from tests.base import VisualTestCase
from tests.dataset import DatasetMixin, Dataset

from yellowbrick.features.rfecv import *
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Fixtures
##########################################################################

@pytest.fixture(scope="class")
def dataset(request):
    """
    Creates a multiclass classification dataset fixture for RFECV
    """
    X, y = make_classification(
        n_samples=600, n_features=15, n_informative=7, n_redundant=4,
        n_repeated=0, n_classes=8, n_clusters_per_class=1, random_state=0
    )

    dataset = Dataset(X, y)
    request.cls.dataset = dataset


##########################################################################
## Test Cases
##########################################################################

@pytest.mark.usefixtures("dataset")
class TestRFECV(VisualTestCase, DatasetMixin):
    """
    Test the RFECV visualizer
    """

    @patch.object(RFECV, 'draw')
    def test_fit(self, mock_draw):
        """
        Assert that fit returns self and creates expected properties with NB
        """
        X, y = self.dataset
        params = (
            "n_features_", "support_", "ranking_",
            "cv_scores_", "rfe_estimator_", "n_feature_subsets_"
        )

        rf = RandomForestClassifier()
        oz = RFECV(rf)
        for param in params:
            assert not hasattr(oz, param)

        # Assert original estimator is wrapped
        assert oz._wrapped is rf

        assert oz.fit(X, y) is oz
        mock_draw.assert_called_once()

        for param in params:
            assert hasattr(oz, param)

        # Assert rfe estimator is now wrapped
        assert oz._wrapped is not rf
        assert oz._wrapped is oz.rfe_estimator_

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_rfecv_classification(self):
        """
        Test image closeness on a classification dataset with an SVM
        """
        cv = ShuffleSplit(3, random_state=21)
        oz = RFECV(SVC(kernel="linear", C=1), cv=cv)
        oz.fit(self.dataset.X, self.dataset.y)
        oz.poof()

        self.assert_images_similar(oz)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    @pytest.mark.filterwarnings('ignore:F-score is ill-defined')
    def test_quick_method(self):
        """
        Test the recv quick method works with LogisticRegression
        """
        cv = ShuffleSplit(2, random_state=14)
        model = LogisticRegression()
        X, y = self.dataset

        ax = rfecv(model, X, y, step=2, cv=cv, scoring='f1_weighted')

        self.assert_images_similar(ax=ax)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_integration(self):
        """
        Test on a real dataset with pandas DataFrame and Series
        """
        df = self.load_pandas("occupancy")

        target = "occupancy"
        features = [
            'temperature', 'relative humidity', 'light', 'C02', 'humidity'
        ]

        X = df[features]
        y = df[target]

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        cv = StratifiedKFold(n_splits=4, random_state=32)
        oz = RFECV(RandomForestClassifier(random_state=83), cv=cv)
        oz.fit(X, y)
        oz.poof()

        self.assert_images_similar(oz)

    def test_invalid_step(self):
        """
        Test step hyperparam validation
        """
        # TODO: parametrize when unittest is removed
        with pytest.raises(YellowbrickValueError, match="step must be >0"):
            oz = RFECV(SVC(kernel="linear"), step=-1)
            oz.fit(self.dataset.X, self.dataset.y)

    def test_rfecv_step(self):
        """
        Test RFECV step=5 with LogisticRegression
        """
        X, y = make_classification(
            n_samples=200, n_features=30, n_informative=18, n_redundant=6,
            n_repeated=0, n_classes=8, n_clusters_per_class=1, random_state=0
        )

        oz = RFECV(LogisticRegression(random_state=32), step=5).fit(X, y)
        assert hasattr(oz, "n_feature_subsets_")
        npt.assert_array_equal(oz.n_feature_subsets_, np.arange(1,35,5))

        oz.finalize()
        tol = 1.75 if sys.platform == "win32" else 0.25
        self.assert_images_similar(oz, tol=tol)