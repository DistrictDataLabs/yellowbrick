# tests.test_regressor.test_influence
# Test the regressor influence visualizers.
#
# Author:   Benjamin Bengfort
# Created:  Sun Jun 09 16:03:31 2019 -0400
#
# Copyright (C) 2019 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_influence.py [fe14cfd] benjamin@bengfort.com $

"""
Test the regressor influence visualizers.
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np
import matplotlib.pyplot as plt

from tests.base import VisualTestCase
from tests.fixtures import Dataset
from sklearn.datasets import make_regression

from yellowbrick.regressor.influence import *
from yellowbrick.datasets import load_concrete

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Fixtures
##########################################################################


@pytest.fixture(scope="class")
def data(request):
    """
    Creates a random regression fixture that has a R2 score below 0.85 and several
    outliers that best demonstrate the effectiveness of influence visualizers.
    """
    X, y = make_regression(
        n_samples=100,
        n_features=14,
        n_informative=6,
        bias=1.2,
        noise=49.8,
        tail_strength=0.6,
        random_state=637,
    )

    request.cls.data = Dataset(X, y)


##########################################################################
## Assertion Helpers
##########################################################################

LEARNED_FIELDS = (
    "distance_",
    "p_values_",
    "influence_threshold_",
    "outlier_percentage_",
)


def assert_not_fitted(oz):
    for field in LEARNED_FIELDS:
        assert not hasattr(oz, field)


def assert_fitted(oz):
    for field in LEARNED_FIELDS:
        assert hasattr(oz, field)


##########################################################################
## Test CooksDistance Visualizer
##########################################################################


@pytest.mark.usefixtures("data")
class TestCooksDistance(VisualTestCase):
    """
    CooksDistance visual test cases
    """

    def test_cooks_distance(self):
        """
        Test image similarity of Cook's Distance on a random dataset
        """
        _, ax = plt.subplots()
        viz = CooksDistance(ax=ax)

        assert_not_fitted(viz)
        assert viz.fit(self.data.X, self.data.y) is viz
        assert_fitted(viz)

        # Test fitted values
        assert viz.distance_.shape == (self.data.X.shape[0],)
        assert viz.p_values_.shape == viz.distance_.shape
        assert 0.0 <= viz.influence_threshold_ <= 4.0
        assert 0.0 <= viz.outlier_percentage_ <= 100.0

        self.assert_images_similar(viz)

    def test_cooks_distance_quickmethod(self):
        """
        Test the cooks_distance quick method on a random dataset
        """
        _, ax = plt.subplots()
        viz = cooks_distance(
            self.data.X,
            self.data.y,
            ax=ax,
            draw_threshold=False,
            linefmt="r-",
            markerfmt="ro",
            show=False
        )

        assert_fitted(viz)
        self.assert_images_similar(viz)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_integration(self):
        """
        Test on the concrete dataset with pandas DataFrame and Series
        """
        data = load_concrete(return_dataset=True)
        X, y = data.to_pandas()

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        _, ax = plt.subplots()
        viz = CooksDistance(ax=ax).fit(X, y)
        assert_fitted(viz)

        assert viz.distance_.sum() == pytest.approx(1.2911900571300652)
        assert viz.p_values_.sum() == pytest.approx(1029.9999525376425)
        assert viz.influence_threshold_ == pytest.approx(0.003883495145631068)
        assert viz.outlier_percentage_ == pytest.approx(7.3786407766990285)

        viz.finalize()
        self.assert_images_similar(viz)

    def test_numpy_integration(self):
        """
        Test on concrete dataset with numpy arrays
        """
        data = load_concrete(return_dataset=True)
        X, y = data.to_numpy()

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

        _, ax = plt.subplots()
        viz = CooksDistance(ax=ax).fit(X, y)
        assert_fitted(viz)

        assert viz.distance_.sum() == pytest.approx(1.2911900571300652)
        assert viz.p_values_.sum() == pytest.approx(1029.9999525376425)
        assert viz.influence_threshold_ == pytest.approx(0.003883495145631068)
        assert viz.outlier_percentage_ == pytest.approx(7.3786407766990285)

        viz.finalize()
        self.assert_images_similar(viz)
