# tests.test_features.test_rankd
# Test the rankd feature analysis visualizers
#
# Author:   Benjamin Bengfort
# Created:  Fri Oct 07 12:19:19 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_rankd.py [7b4350a] nathan.danielsen@gmail.com $

"""
Test the Rankd feature analysis visualizers
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np
import numpy.testing as npt

from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase

from yellowbrick.datasets import load_occupancy
from yellowbrick.features.rankd import *
from yellowbrick.features.rankd import kendalltau
from yellowbrick.features.rankd import RankDBase
from sklearn.datasets import make_regression

from yellowbrick.exceptions import YellowbrickValueError

try:
    import pandas as pd
except ImportError:
    pd = None


@pytest.fixture(scope="class")
def dataset(request):
    """
    Creates a dataset with 6 gaussian features and 2 categorical features
    for testing the RankD ranking algorithms. The gaussian features have
    different correlations with respect to each other, including strong
    positive and negative correlation and no correlation at all.
    """
    X, _ = make_regression(
        n_samples=100,
        n_features=6,
        effective_rank=2,
        tail_strength=0,
        n_informative=2,
        noise=0.45,
        random_state=27,
    )

    rand = np.random.RandomState(seed=27)
    request.cls.dataset = np.concatenate((X, rand.binomial(1, 0.6, (100, 2))), axis=1)


##########################################################################
## Kendall-Tau Tests
##########################################################################


@pytest.mark.usefixtures("dataset")
class TestKendallTau(object):
    """
    Test the Kendall-Tau correlation metric
    """

    def test_kendalltau(self):
        """
        Test results returned match expectations
        """
        expected = np.array(
            [
                [
                    1.0,
                    -0.68,
                    -0.57454545,
                    0.49858586,
                    0.07555556,
                    -0.05858586,
                    0.02387848,
                    0.11357219,
                ],
                [
                    -0.68,
                    1.0,
                    0.58666667,
                    -0.69090909,
                    -0.22262626,
                    -0.17171717,
                    -0.05059964,
                    -0.12397575,
                ],
                [
                    -0.57454545,
                    0.58666667,
                    1.0,
                    -0.61050505,
                    0.18909091,
                    0.07515152,
                    0.00341121,
                    -0.0638663,
                ],
                [
                    0.49858586,
                    -0.69090909,
                    -0.61050505,
                    1.0,
                    0.11070707,
                    0.3030303,
                    0.03013237,
                    0.07542581,
                ],
                [
                    0.07555556,
                    -0.22262626,
                    0.18909091,
                    0.11070707,
                    1.0,
                    0.4610101,
                    0.01648752,
                    0.05982047,
                ],
                [
                    -0.05858586,
                    -0.17171717,
                    0.07515152,
                    0.3030303,
                    0.4610101,
                    1.0,
                    0.03695479,
                    -0.02398599,
                ],
                [
                    0.02387848,
                    -0.05059964,
                    0.00341121,
                    0.03013237,
                    0.01648752,
                    0.03695479,
                    1.0,
                    0.18298883,
                ],
                [
                    0.11357219,
                    -0.12397575,
                    -0.0638663,
                    0.07542581,
                    0.05982047,
                    -0.02398599,
                    0.18298883,
                    1.0,
                ],
            ]
        )
        npt.assert_almost_equal(expected, kendalltau(self.dataset))

    def test_kendalltau_shape(self):
        """
        Assert that a square correlation matrix is returned
        """
        corr = kendalltau(self.dataset)
        assert corr.shape[0] == corr.shape[1]

        for (i, j), val in np.ndenumerate(corr):
            assert corr[j][i] == pytest.approx(val)

    def test_kendalltau_1D(self):
        """
        Assert that a 2D matrix is required as input
        """
        with pytest.raises(IndexError, match="tuple index out of range"):
            X = 0.1 * np.arange(10)
            kendalltau(X)


##########################################################################
## RankDBase Tests
##########################################################################


@pytest.mark.usefixtures("dataset")
class TestRankDBase(VisualTestCase):
    """
    Test the RankDBase Visualizer
    """

    def test_rankdbase_unknown_algorithm(self):
        """
        Assert that unknown algorithms raise an exception
        """
        with pytest.raises(
            YellowbrickValueError, match=".* is unrecognized ranking method"
        ) as e:
            oz = RankDBase(algorithm="unknown")
            oz.fit_transform(self.dataset)
            assert str(e.value) == "'unknown' is unrecognized ranking method"


##########################################################################
## Rank1D Base Tests
##########################################################################


@pytest.mark.usefixtures("dataset")
class TestRank1D(VisualTestCase):
    """
    Test the Rank1D visualizer
    """

    def test_rank1d_shapiro(self):
        """
        Test Rank1D using shapiro metric
        """
        oz = Rank1D(algorithm="shapiro")
        npt.assert_array_equal(oz.fit_transform(self.dataset), self.dataset)

        # Check Ranking
        expected = np.array(
            [
                0.985617,
                0.992236,
                0.982354,
                0.984898,
                0.978514,
                0.990372,
                0.636401,
                0.624511,
            ]
        )

        assert hasattr(oz, "ranks_")
        assert oz.ranks_.shape == (self.dataset.shape[1],)
        npt.assert_array_almost_equal(oz.ranks_, expected)

        # Image similarity comparison
        oz.finalize()
        self.assert_images_similar(oz)

    def test_rank1d_orientation(self):
        """
        Test Rank1D using vertical orientation
        """
        oz = Rank1D(orient="v")
        npt.assert_array_equal(oz.fit_transform(self.dataset), self.dataset)

        # Image similarity comparison
        oz.finalize()
        self.assert_images_similar(oz)

    @pytest.mark.filterwarnings("ignore:p-value")
    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_rank1d_integrated_pandas(self):
        """
        Test Rank1D on occupancy dataset with pandas DataFrame and Series
        """
        data = load_occupancy(return_dataset=True)
        X, y = data.to_pandas()
        features = data.meta["features"]

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        # Test the visualizer
        oz = Rank1D(features=features, show_feature_names=True)
        assert oz.fit(X, y) is oz
        assert oz.transform(X) is X

        # Image similarity testing
        oz.finalize()
        self.assert_images_similar(oz)

    @pytest.mark.filterwarnings("ignore:p-value")
    def test_rank1d_integrated_numpy(self):
        """
        Test Rank1D on occupancy dataset with default numpy data structures
        """
        data = load_occupancy(return_dataset=True)
        X, y = data.to_numpy()
        features = data.meta["features"]

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

        # Test the visualizer
        oz = Rank1D(features=features, show_feature_names=True)
        assert oz.fit(X, y) is oz
        assert oz.transform(X) is X

        # Image similarity testing
        oz.finalize()
        self.assert_images_similar(oz)


##########################################################################
## Rank2D Test Cases
##########################################################################


@pytest.mark.usefixtures("dataset")
class TestRank2D(VisualTestCase):
    """
    Test the Rank2D visualizer
    """

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_rank2d_pearson(self):
        """
        Test Rank2D using pearson metric
        """
        oz = Rank2D(algorithm="pearson")
        npt.assert_array_equal(oz.fit_transform(self.dataset), self.dataset)

        # Check Ranking
        expected = np.array(
            [
                [
                    1.0,
                    -0.86937243,
                    -0.77884764,
                    0.71424708,
                    0.10836854,
                    -0.11550965,
                    0.04494811,
                    0.1725682,
                ],
                [
                    -0.86937243,
                    1.0,
                    0.80436327,
                    -0.9086706,
                    -0.31117192,
                    -0.26313947,
                    -0.0711807,
                    -0.16924862,
                ],
                [
                    -0.77884764,
                    0.80436327,
                    1.0,
                    -0.85520468,
                    0.30940711,
                    0.10634903,
                    -0.02485686,
                    -0.10230028,
                ],
                [
                    0.71424708,
                    -0.9086706,
                    -0.85520468,
                    1.0,
                    0.12537213,
                    0.41306822,
                    0.04704408,
                    0.1031842,
                ],
                [
                    0.10836854,
                    -0.31117192,
                    0.30940711,
                    0.12537213,
                    1.0,
                    0.671111,
                    0.06777278,
                    0.09513859,
                ],
                [
                    -0.11550965,
                    -0.26313947,
                    0.10634903,
                    0.41306822,
                    0.671111,
                    1.0,
                    0.04684117,
                    -0.01072631,
                ],
                [
                    0.04494811,
                    -0.0711807,
                    -0.02485686,
                    0.04704408,
                    0.06777278,
                    0.04684117,
                    1.0,
                    0.18298883,
                ],
                [
                    0.1725682,
                    -0.16924862,
                    -0.10230028,
                    0.1031842,
                    0.09513859,
                    -0.01072631,
                    0.18298883,
                    1.0,
                ],
            ]
        )

        assert hasattr(oz, "ranks_")
        assert oz.ranks_.shape == (self.dataset.shape[1], self.dataset.shape[1])
        npt.assert_array_almost_equal(oz.ranks_, expected)

        # Image similarity comparision
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_rank2d_covariance(self):
        """
        Test Rank2D using covariance metric
        """
        oz = Rank2D(algorithm="covariance")
        npt.assert_array_equal(oz.fit_transform(self.dataset), self.dataset)

        # Check Ranking
        expected = np.array(
            [
                [
                    4.09266931e-03,
                    -1.41062431e-03,
                    -2.26778429e-03,
                    3.13507202e-03,
                    2.21273274e-04,
                    -5.05566875e-04,
                    1.44499782e-03,
                    5.45713163e-03,
                ],
                [
                    -1.41062431e-03,
                    6.43286363e-04,
                    9.28539346e-04,
                    -1.58126396e-03,
                    -2.51898163e-04,
                    -4.56609749e-04,
                    -9.07228811e-04,
                    -2.12191333e-03,
                ],
                [
                    -2.26778429e-03,
                    9.28539346e-04,
                    2.07153281e-03,
                    -2.67061756e-03,
                    4.49467833e-04,
                    3.31158917e-04,
                    -5.68518509e-04,
                    -2.30156415e-03,
                ],
                [
                    3.13507202e-03,
                    -1.58126396e-03,
                    -2.67061756e-03,
                    4.70751209e-03,
                    2.74548546e-04,
                    1.93898526e-03,
                    1.62200836e-03,
                    3.49952628e-03,
                ],
                [
                    2.21273274e-04,
                    -2.51898163e-04,
                    4.49467833e-04,
                    2.74548546e-04,
                    1.01869657e-03,
                    1.46545939e-03,
                    1.08700151e-03,
                    1.50099581e-03,
                ],
                [
                    -5.05566875e-04,
                    -4.56609749e-04,
                    3.31158917e-04,
                    1.93898526e-03,
                    1.46545939e-03,
                    4.68073451e-03,
                    1.61041253e-03,
                    -3.62750059e-04,
                ],
                [
                    1.44499782e-03,
                    -9.07228811e-04,
                    -5.68518509e-04,
                    1.62200836e-03,
                    1.08700151e-03,
                    1.61041253e-03,
                    2.52525253e-01,
                    4.54545455e-02,
                ],
                [
                    5.45713163e-03,
                    -2.12191333e-03,
                    -2.30156415e-03,
                    3.49952628e-03,
                    1.50099581e-03,
                    -3.62750059e-04,
                    4.54545455e-02,
                    2.44343434e-01,
                ],
            ]
        )

        assert hasattr(oz, "ranks_")
        assert oz.ranks_.shape == (self.dataset.shape[1], self.dataset.shape[1])
        npt.assert_array_almost_equal(oz.ranks_, expected)

        # Image similarity comparision
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_rank2d_spearman(self):
        """
        Test Rank2D using spearman metric
        """
        oz = Rank2D(algorithm="spearman")
        npt.assert_array_equal(oz.fit_transform(self.dataset), self.dataset)

        # Check Ranking
        expected = np.array(
            [
                [
                    1.0,
                    -0.86889889,
                    -0.77551755,
                    0.68520852,
                    0.11369937,
                    -0.09489349,
                    0.02909991,
                    0.13840665,
                ],
                [
                    -0.86889889,
                    1.0,
                    0.78232223,
                    -0.87065107,
                    -0.33450945,
                    -0.25244524,
                    -0.06166409,
                    -0.15108512,
                ],
                [
                    -0.77551755,
                    0.78232223,
                    1.0,
                    -0.81636964,
                    0.26846685,
                    0.10348635,
                    0.00415713,
                    -0.07783173,
                ],
                [
                    0.68520852,
                    -0.87065107,
                    -0.81636964,
                    1.0,
                    0.16316832,
                    0.45167717,
                    0.03672131,
                    0.09191892,
                ],
                [
                    0.11369937,
                    -0.33450945,
                    0.26846685,
                    0.16316832,
                    1.0,
                    0.63986799,
                    0.02009279,
                    0.07290121,
                ],
                [
                    -0.09489349,
                    -0.25244524,
                    0.10348635,
                    0.45167717,
                    0.63986799,
                    1.0,
                    0.04503557,
                    -0.02923092,
                ],
                [
                    0.02909991,
                    -0.06166409,
                    0.00415713,
                    0.03672131,
                    0.02009279,
                    0.04503557,
                    1.0,
                    0.18298883,
                ],
                [
                    0.13840665,
                    -0.15108512,
                    -0.07783173,
                    0.09191892,
                    0.07290121,
                    -0.02923092,
                    0.18298883,
                    1.0,
                ],
            ]
        )

        assert hasattr(oz, "ranks_")
        assert oz.ranks_.shape == (self.dataset.shape[1], self.dataset.shape[1])
        npt.assert_array_almost_equal(oz.ranks_, expected)

        # Image similarity comparision
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_rank2d_kendalltau(self):
        """
        Test Rank2D using kendalltau metric
        """
        oz = Rank2D(algorithm="kendalltau")
        npt.assert_array_equal(oz.fit_transform(self.dataset), self.dataset)

        # Check Ranking
        expected = np.array(
            [
                [
                    1.0,
                    -0.68,
                    -0.57454545,
                    0.49858586,
                    0.07555556,
                    -0.05858586,
                    0.02387848,
                    0.11357219,
                ],
                [
                    -0.68,
                    1.0,
                    0.58666667,
                    -0.69090909,
                    -0.22262626,
                    -0.17171717,
                    -0.05059964,
                    -0.12397575,
                ],
                [
                    -0.57454545,
                    0.58666667,
                    1.0,
                    -0.61050505,
                    0.18909091,
                    0.07515152,
                    0.00341121,
                    -0.0638663,
                ],
                [
                    0.49858586,
                    -0.69090909,
                    -0.61050505,
                    1.0,
                    0.11070707,
                    0.3030303,
                    0.03013237,
                    0.07542581,
                ],
                [
                    0.07555556,
                    -0.22262626,
                    0.18909091,
                    0.11070707,
                    1.0,
                    0.4610101,
                    0.01648752,
                    0.05982047,
                ],
                [
                    -0.05858586,
                    -0.17171717,
                    0.07515152,
                    0.3030303,
                    0.4610101,
                    1.0,
                    0.03695479,
                    -0.02398599,
                ],
                [
                    0.02387848,
                    -0.05059964,
                    0.00341121,
                    0.03013237,
                    0.01648752,
                    0.03695479,
                    1.0,
                    0.18298883,
                ],
                [
                    0.11357219,
                    -0.12397575,
                    -0.0638663,
                    0.07542581,
                    0.05982047,
                    -0.02398599,
                    0.18298883,
                    1.0,
                ],
            ]
        )

        assert hasattr(oz, "ranks_")
        assert oz.ranks_.shape == (self.dataset.shape[1], self.dataset.shape[1])
        npt.assert_array_almost_equal(oz.ranks_, expected)

        # Image similarity comparision
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_rank2d_integrated_pandas(self):
        """
        Test Rank2D on occupancy dataset with pandas DataFrame and Series
        """
        data = load_occupancy(return_dataset=True)
        X, y = data.to_pandas()
        features = data.meta["features"]

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        # Test the visualizer
        oz = Rank2D(features=features, show_feature_names=True)
        assert oz.fit(X, y) is oz
        assert oz.transform(X) is X
        oz.finalize()

        # Image similarity testing
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_rank2d_integrated_numpy(self):
        """
        Test Rank2D on occupancy dataset with numpy ndarray
        """
        data = load_occupancy(return_dataset=True)
        X, y = data.to_numpy()
        features = data.meta["features"]

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

        # Test the visualizer
        oz = Rank2D(features=features, show_feature_names=True)
        assert oz.fit(X, y) is oz
        assert oz.transform(X) is X
        oz.finalize()

        # Image similarity testing
        self.assert_images_similar(oz, tol=0.1)
