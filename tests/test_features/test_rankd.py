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

from yellowbrick.features.rankd import RankDBase
from yellowbrick.features.rankd import kendalltau
from yellowbrick.features.rankd import Rank1D, rank1d
from yellowbrick.features.rankd import Rank2D, rank2d
from yellowbrick.exceptions import YellowbrickValueError
from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase
from yellowbrick.datasets import load_occupancy, load_credit, load_energy

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Kendall-Tau Tests
##########################################################################


class TestKendallTau(object):
    """
    Test the Kendall-Tau correlation metric
    """

    def test_kendalltau(self):
        """
        Test results returned match expectations
        """
        X, _ = load_energy(return_dataset=True).to_numpy()

        expected = np.array(
            [
                [1.0, -1.0, -0.2724275, -0.7361443, 0.7385489, 0.0, 0.0, 0.0],
                [-1.0, 1.0, 0.2724275, 0.7361443, -0.7385489, 0.0, 0.0, 0.0],
                [-0.2724275, 0.2724275, 1.0, -0.15192004, 0.19528337, 0.0, 0.0, 0.0],
                [-0.73614431, 0.73614431, -0.15192004, 1.0, -0.87518995, 0.0, 0.0, 0.0],
                [0.73854895, -0.73854895, 0.19528337, -0.87518995, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.15430335],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15430335, 1.0],
            ]
        )
        actual = kendalltau(X)
        npt.assert_almost_equal(expected, actual)

    def test_kendalltau_shape(self):
        """
        Assert that a square correlation matrix is returned
        """
        X, _ = load_energy(return_dataset=True).to_numpy()
        corr = kendalltau(X)
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


class TestRankDBase(VisualTestCase):
    """
    Test the RankDBase Visualizer
    """

    def test_rankdbase_unknown_algorithm(self):
        """
        Assert that unknown algorithms raise an exception
        """
        X, _ = load_energy(return_dataset=True).to_numpy()
        with pytest.raises(
            YellowbrickValueError, match=".* is unrecognized ranking method"
        ) as e:
            oz = RankDBase(algorithm="unknown")
            oz.fit_transform(X)
            assert str(e.value) == "'unknown' is unrecognized ranking method"


##########################################################################
## Rank1D Base Tests
##########################################################################


class TestRank1D(VisualTestCase):
    """
    Test the Rank1D visualizer
    """

    def test_rank1d_unknown_algorithm(self):
        """
        Test that an error is raised for Rank1D with an unknown algorithm
        """
        X, _ = load_energy()
        msg = "'oscar' is unrecognized ranking method"
        with pytest.raises(YellowbrickValueError, match=msg):
            Rank1D(algorithm="Oscar").transform(X)

    def test_rank1d_shapiro(self):
        """
        Test Rank1D using shapiro metric
        """
        X, _ = load_energy(return_dataset=True).to_numpy()
        oz = Rank1D(algorithm="shapiro")
        npt.assert_array_equal(oz.fit_transform(X), X)

        # Check Ranking
        expected = np.array(
            [
                0.93340671,
                0.94967198,
                0.92689574,
                0.7459445,
                0.63657606,
                0.85603625,
                0.84349269,
                0.91551381,
            ]
        )

        assert hasattr(oz, "ranks_")
        assert oz.ranks_.shape == (X.shape[1],)
        npt.assert_array_almost_equal(oz.ranks_, expected)

        # Image similarity comparison
        oz.finalize()
        self.assert_images_similar(oz)

    def test_rank1d_vertical(self):
        """
        Test Rank1D using vertical orientation
        """
        X, _ = load_energy(return_dataset=True).to_numpy()
        oz = Rank1D(orient="v")
        npt.assert_array_equal(oz.fit_transform(X), X)

        # Image similarity comparison
        oz.finalize()
        self.assert_images_similar(oz)

    def test_rank1d_horizontal(self):
        """
        Test Rank1D using horizontal orientation
        """
        X, _ = load_energy(return_dataset=True).to_numpy()
        oz = Rank1D(orient="h")
        npt.assert_array_equal(oz.fit_transform(X), X)

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

    def test_rank1d_quick_method(self):
        """
        Test Rank1d quick method
        """
        X, y = load_credit()
        viz = rank1d(X, y, show=False)

        assert isinstance(viz, Rank1D)
        self.assert_images_similar(viz, tol=0.1)


##########################################################################
## Rank2D Test Cases
##########################################################################


class TestRank2D(VisualTestCase):
    """
    Test the Rank2D visualizer
    """
    def test_rank2d_unknown_algorithm(self):
        """
        Test that an error is raised for Rank2D with an unknown algorithm
        """
        X, _ = load_energy()
        msg = "'oscar' is unrecognized ranking method"
        with pytest.raises(YellowbrickValueError, match=msg):
            Rank2D(algorithm="Oscar").transform(X)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_rank2d_pearson(self):
        """
        Test Rank2D using pearson metric
        """
        X, _ = load_energy(return_dataset=True).to_numpy()
        oz = Rank2D(algorithm="pearson")
        npt.assert_array_equal(oz.fit_transform(X), X)

        # Check Ranking
        expected = np.array(
            [
                [
                    1.00000000e00,
                    -9.91901462e-01,
                    -2.03781680e-01,
                    -8.68823408e-01,
                    8.27747317e-01,
                    0.00000000e00,
                    1.11706815e-16,
                    -1.12935670e-16,
                ],
                [
                    -9.91901462e-01,
                    1.00000000e00,
                    1.95501633e-01,
                    8.80719517e-01,
                    -8.58147673e-01,
                    0.00000000e00,
                    -2.26567708e-16,
                    -3.55861251e-16,
                ],
                [
                    -2.03781680e-01,
                    1.95501633e-01,
                    1.00000000e00,
                    -2.92316466e-01,
                    2.80975743e-01,
                    0.00000000e00,
                    7.87010445e-18,
                    0.00000000e00,
                ],
                [
                    -8.68823408e-01,
                    8.80719517e-01,
                    -2.92316466e-01,
                    1.00000000e00,
                    -9.72512237e-01,
                    0.00000000e00,
                    -3.27553310e-16,
                    2.20057668e-16,
                ],
                [
                    8.27747317e-01,
                    -8.58147673e-01,
                    2.80975743e-01,
                    -9.72512237e-01,
                    1.00000000e00,
                    0.00000000e00,
                    -1.24094525e-18,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    1.00000000e00,
                    -2.42798319e-19,
                    0.00000000e00,
                ],
                [
                    1.11706815e-16,
                    -2.26567708e-16,
                    7.87010445e-18,
                    -3.27553310e-16,
                    -1.24094525e-18,
                    -2.42798319e-19,
                    1.00000000e00,
                    2.12964221e-01,
                ],
                [
                    -1.12935670e-16,
                    -3.55861251e-16,
                    0.00000000e00,
                    2.20057668e-16,
                    0.00000000e00,
                    0.00000000e00,
                    2.12964221e-01,
                    1.00000000e00,
                ],
            ]
        )

        assert hasattr(oz, "ranks_")
        assert oz.ranks_.shape == (X.shape[1], X.shape[1])
        npt.assert_array_almost_equal(oz.ranks_, expected)

        # Image similarity comparision
        oz.finalize()
        # Travis Python 3.6 images not close (RMS 0.112)
        self.assert_images_similar(oz, tol=0.5)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_rank2d_covariance(self):
        """
        Test Rank2D using covariance metric
        """
        X, _ = load_energy(return_dataset=True).to_numpy()
        oz = Rank2D(algorithm="covariance")
        npt.assert_array_equal(oz.fit_transform(X), X)

        # Check Ranking
        expected = np.array(
            [
                [
                    1.11888744e-02,
                    -9.24206867e00,
                    -9.40391134e-01,
                    -4.15083877e00,
                    1.53324641e-01,
                    0.00000000e00,
                    1.57414282e-18,
                    -1.85278419e-17,
                ],
                [
                    -9.24206867e00,
                    7.75916384e03,
                    7.51290743e02,
                    3.50393655e03,
                    -1.32370274e02,
                    0.00000000e00,
                    -2.65874531e-15,
                    -4.86170571e-14,
                ],
                [
                    -9.40391134e-01,
                    7.51290743e02,
                    1.90326988e03,
                    -5.75989570e02,
                    2.14654498e01,
                    0.00000000e00,
                    4.57406096e-17,
                    0.00000000e00,
                ],
                [
                    -4.15083877e00,
                    3.50393655e03,
                    -5.75989570e02,
                    2.03996306e03,
                    -7.69178618e01,
                    0.00000000e00,
                    -1.97089918e-15,
                    1.54151644e-14,
                ],
                [
                    1.53324641e-01,
                    -1.32370274e02,
                    2.14654498e01,
                    -7.69178618e01,
                    3.06649283e00,
                    0.00000000e00,
                    -2.89497529e-19,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    1.25162973e00,
                    -3.61871912e-20,
                    0.00000000e00,
                ],
                [
                    1.57414282e-18,
                    -2.65874531e-15,
                    4.57406096e-17,
                    -1.97089918e-15,
                    -2.89497529e-19,
                    -3.61871912e-20,
                    1.77477184e-02,
                    4.40026076e-02,
                ],
                [
                    -1.85278419e-17,
                    -4.86170571e-14,
                    0.00000000e00,
                    1.54151644e-14,
                    0.00000000e00,
                    0.00000000e00,
                    4.40026076e-02,
                    2.40547588e00,
                ],
            ]
        )

        assert hasattr(oz, "ranks_")
        assert oz.ranks_.shape == (X.shape[1], X.shape[1])
        npt.assert_array_almost_equal(oz.ranks_, expected, decimal=5)

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
        X, _ = load_energy(return_dataset=True).to_numpy()
        oz = Rank2D(algorithm="spearman")
        npt.assert_array_equal(oz.fit_transform(X), X)

        # Check Ranking
        expected = np.array(
            [
                [1.0, -1.0, -0.25580533, -0.8708862, 0.86904819, 0.0, 0.0, 0.0],
                [-1.0, 1.0, 0.25580533, 0.8708862, -0.86904819, 0.0, 0.0, 0.0],
                [-0.25580533, 0.25580533, 1.0, -0.19345677, 0.22076336, 0.0, 0.0, 0.0],
                [-0.8708862, 0.8708862, -0.19345677, 1.0, -0.93704257, 0.0, 0.0, 0.0],
                [0.86904819, -0.86904819, 0.22076336, -0.93704257, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.18759162],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18759162, 1.0],
            ]
        )

        assert hasattr(oz, "ranks_")
        assert oz.ranks_.shape == (X.shape[1], X.shape[1])
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
        X, _ = load_energy(return_dataset=True).to_numpy()
        oz = Rank2D(algorithm="kendalltau")
        npt.assert_array_equal(oz.fit_transform(X), X)

        # Check Ranking
        expected = np.array(
            [
                [1.0, -1.0, -0.2724275, -0.73614431, 0.73854895, 0.0, 0.0, 0.0],
                [-1.0, 1.0, 0.2724275, 0.73614431, -0.73854895, 0.0, 0.0, 0.0],
                [-0.2724275, 0.2724275, 1.0, -0.15192004, 0.19528337, 0.0, 0.0, 0.0],
                [-0.73614431, 0.73614431, -0.15192004, 1.0, -0.87518995, 0.0, 0.0, 0.0],
                [0.73854895, -0.73854895, 0.19528337, -0.87518995, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.15430335],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15430335, 1.0],
            ]
        )

        assert hasattr(oz, "ranks_")
        assert oz.ranks_.shape == (X.shape[1], X.shape[1])
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

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_rank2d_quick_method(self):
        """
        Test Rank2D quick method
        """
        X, y = load_occupancy()
        oz = rank2d(X, y, algorithm="spearman", colormap="RdYlGn_r")

        assert isinstance(oz, Rank2D)
        self.assert_images_similar(oz, tol=0.1)
