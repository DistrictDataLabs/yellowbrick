# tests.test_features.test_rankd
# Test the rankd feature analysis visualizers
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Oct 07 12:19:19 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_rankd.py [01d5996] benjamin@bengfort.com $

"""
Test the Rankd feature analysis visualizers
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest
import numpy as np
import numpy.testing as npt

from tests.base import VisualTestCase

from yellowbrick.datasets import load_occupancy, load_energy
from yellowbrick.features.rankd import *
from yellowbrick.features.rankd import kendalltau
from yellowbrick.features.rankd import RankDBase

from yellowbrick.exceptions import YellowbrickValueError

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

        expected = np.array([[1., -1., -0.2724275, -0.73614431, 0.73854895, 0., 0., 0.],
                             [-1., 1., 0.2724275, 0.73614431, -0.73854895, 0., 0., 0.],
                             [-0.2724275, 0.2724275, 1., -0.15192004, 0.19528337, 0., 0., 0.],
                             [-0.73614431, 0.73614431, -0.15192004, 1., -0.87518995, 0., 0., 0.],
                             [0.73854895, -0.73854895, 0.19528337, -0.87518995, 1., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 1., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 1., 0.15430335],
                             [0., 0., 0., 0., 0., 0., 0.15430335, 1.]])

        X, y = load_energy()
        npt.assert_almost_equal(expected, kendalltau(X.to_numpy()))

    def test_kendalltau_shape(self):
        """
        Assert that a square correlation matrix is returned
        """
        X, y = load_energy()
        corr = kendalltau(X.to_numpy())
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
        X, y = load_energy()
        with pytest.raises(YellowbrickValueError,
                           match='.* is unrecognized ranking method') as e:
            oz = RankDBase(algorithm='unknown')
            oz.fit_transform(X, y)
            assert str(e.value) == "'unknown' is unrecognized ranking method"


##########################################################################
## Rank1D Base Tests
##########################################################################

class TestRank1D(VisualTestCase):
    """
    Test the Rank1D visualizer
    """

    def test_rank1d_shapiro(self):
        """
        Test Rank1D using shapiro metric
        """
        X, y = load_energy()
        oz = Rank1D(algorithm='shapiro')
        npt.assert_array_equal(oz.fit_transform(X, y), X.to_numpy())

        # Check Ranking
        expected = np.array([0.93340671, 0.94967198, 0.92689574, 0.7459445,
                             0.63657606, 0.85603625, 0.84349269, 0.91551381])

        assert hasattr(oz, 'ranks_')
        assert oz.ranks_.shape == (X.shape[1],)
        npt.assert_array_almost_equal(oz.ranks_, expected)

        # Image similarity comparison
        oz.finalize()
        self.assert_images_similar(oz)

    def test_rank1d_orientation(self):
        """
        Test Rank1D using vertical orientation
        """
        X, y = load_energy()
        oz = Rank1D(orient='v')
        npt.assert_array_equal(oz.fit_transform(X, y), X.to_numpy())

        # Image similarity comparison
        oz.finalize()
        self.assert_images_similar(oz)

        with pytest.raises(YellowbrickValueError,
                           match="Orientation must be 'h' or 'v'"):
            oz = Rank1D(orient='x')
            oz.fit_transform_poof(X, y)

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

class TestRank2D(VisualTestCase):
    """
    Test the Rank2D visualizer
    """

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_rank2d_pearson(self):
        """
        Test Rank2D using pearson metric
        """
        X, y = load_energy()
        oz = Rank2D(algorithm='pearson')
        npt.assert_array_equal(oz.fit_transform(X, y), X.to_numpy())

        # Check Ranking
        expected = np.array([[1.000000000000000000e+00, -9.919014616138865925e-01,
                              -2.037816803210378835e-01, -8.688234077044781900e-01,
                              8.277473168384291702e-01, 0.000000000000000000e+00,
                              1.014426274964113587e-16, 1.454708484130729174e-16],
                             [-9.919014616138867035e-01, 9.999999999999998890e-01,
                              1.955016327893900618e-01, 8.807195166848427181e-01,
                              -8.581476730290201749e-01, 0.000000000000000000e+00,
                              -3.503655343914980818e-18, -5.954980011880410679e-16],
                             [-2.037816803210378835e-01, 1.955016327893900618e-01,
                              1.000000000000000000e+00, -2.923164661948912668e-01,
                              2.809757434745081550e-01, 0.000000000000000000e+00,
                              7.633503211062052797e-18, 0.000000000000000000e+00],
                             [-8.688234077044783010e-01, 8.807195166848427181e-01,
                              -2.923164661948913223e-01, 1.000000000000000000e+00,
                              -9.725122370185886878e-01, 0.000000000000000000e+00,
                              -8.506972454574112353e-17, 2.314969641798836556e-16],
                             [8.277473168384291702e-01, -8.581476730290202859e-01,
                              2.809757434745081550e-01, -9.725122370185886878e-01,
                              1.000000000000000000e+00, 0.000000000000000000e+00,
                              1.589961099232681214e-18, 0.000000000000000000e+00],
                             [0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 1.000000000000000000e+00,
                              -2.427983189878238470e-19, 0.000000000000000000e+00],
                             [1.014426274964113587e-16, -3.503655343914981589e-18,
                              7.633503211062052797e-18, -8.506972454574112353e-17,
                              1.589961099232681214e-18, -2.427983189878238951e-19,
                              1.000000000000000000e+00, 2.129642207571907364e-01],
                             [1.454708484130729420e-16, -5.954980011880410679e-16,
                              0.000000000000000000e+00, 2.314969641798836063e-16,
                              0.000000000000000000e+00, 0.000000000000000000e+00,
                              2.129642207571907087e-01, 1.000000000000000000e+00]])

        assert hasattr(oz, 'ranks_')
        assert oz.ranks_.shape == (X.shape[1], X.shape[1])
        npt.assert_array_almost_equal(oz.ranks_, expected)

        # Image similarity comparision
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_rank2d_covariance(self):
        """
        Test Rank2D using covariance metric
        """
        X, y = load_energy()
        oz = Rank2D(algorithm='covariance')
        npt.assert_array_equal(oz.fit_transform(X, y), X.to_numpy())

        # Check Ranking
        expected = np.array([[1.118887440243372666e-02, -9.242068665797489757e+00,
                              -9.403911342894379910e-01, -4.150838765754023996e+00,
                              1.533246414602348617e-01, 0.000000000000000000e+00,
                              1.429502612182775646e-18, 2.386545256650230467e-17],
                             [-9.242068665797489757e+00, 7.759163841807911012e+03,
                              7.512907431551495847e+02, 3.503936549326376280e+03,
                              -1.323702737940027419e+02, 0.000000000000000000e+00,
                              -4.111498191015288127e-17, -8.135575365639113572e-14],
                             [-9.403911342894379910e-01, 7.512907431551495847e+02,
                              1.903269882659713176e+03, -5.759895697522815681e+02,
                              2.146544980443285411e+01, 0.000000000000000000e+00,
                              4.436549635562065831e-17, 0.000000000000000000e+00],
                             [-4.150838765754023996e+00, 3.503936549326376280e+03,
                              -5.759895697522815681e+02, 2.039963059539323467e+03,
                              -7.691786179921777489e+01, 0.000000000000000000e+00,
                              -5.118673665428626920e-16, 1.621649359775821774e-14],
                             [1.533246414602348617e-01, -1.323702737940027419e+02,
                              2.146544980443285411e+01, -7.691786179921777489e+01,
                              3.066492829204693571e+00, 0.000000000000000000e+00,
                              3.709187093353277272e-19, 0.000000000000000000e+00],
                             [0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 1.251629726205997439e+00,
                              -3.618719115466611826e-20, 0.000000000000000000e+00],
                             [1.429502612182775646e-18, -4.111498191015288127e-17,
                              4.436549635562065831e-17, -5.118673665428626920e-16,
                              3.709187093353277272e-19, -3.618719115466611826e-20,
                              1.774771838331158300e-02, 4.400260756192969636e-02],
                             [2.386545256650230467e-17, -8.135575365639113572e-14,
                              0.000000000000000000e+00, 1.621649359775821774e-14,
                              0.000000000000000000e+00, 0.000000000000000000e+00,
                              4.400260756192969636e-02, 2.405475880052151183e+00]])

        assert hasattr(oz, 'ranks_')
        assert oz.ranks_.shape == (X.shape[1], X.shape[1])
        npt.assert_array_almost_equal(oz.ranks_, expected)

        # Image similarity comparision
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_rank2d_spearman(self):
        """
        Test Rank2D using spearman metric
        """
        X, y = load_energy()
        oz = Rank2D(algorithm='spearman')
        npt.assert_array_equal(oz.fit_transform(X, y), X.to_numpy())

        # Check Ranking
        expected = np.array([[1.000000000000000000e+00, -1.000000000000000000e+00,
                              -2.558053335750960500e-01, -8.708862019434340240e-01,
                              8.690481892534817066e-01, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00],
                             [-1.000000000000000000e+00, 1.000000000000000000e+00,
                              2.558053335750960500e-01, 8.708862019434340240e-01,
                              -8.690481892534817066e-01, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00],
                             [-2.558053335750960500e-01, 2.558053335750960500e-01,
                              1.000000000000000000e+00, -1.934567733944697887e-01,
                              2.207633622090921510e-01, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00],
                             [-8.708862019434340240e-01, 8.708862019434340240e-01,
                              -1.934567733944698165e-01, 1.000000000000000000e+00,
                              -9.370425713316365979e-01, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00],
                             [8.690481892534817066e-01, -8.690481892534817066e-01,
                              2.207633622090921510e-01, -9.370425713316364869e-01,
                              1.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00],
                             [0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 1.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00],
                             [0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00,
                              1.000000000000000000e+00, 1.875916198442167115e-01],
                             [0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00,
                              1.875916198442167393e-01, 9.999999999999998890e-01]])

        assert hasattr(oz, 'ranks_')
        assert oz.ranks_.shape == (X.shape[1], X.shape[1])
        npt.assert_array_almost_equal(oz.ranks_, expected)

        # Image similarity comparision
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_rank2d_kendalltau(self):
        """
        Test Rank2D using kendalltau metric
        """
        X, y = load_energy()
        oz = Rank2D(algorithm='kendalltau')
        npt.assert_array_equal(oz.fit_transform(X), X.to_numpy())

        # Check Ranking
        expected = np.array([[9.999999999999997780e-01, -9.999999999999997780e-01,
                              -2.724275017472889693e-01, -7.361443106917049395e-01,
                              7.385489458759963988e-01, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00],
                             [-9.999999999999997780e-01, 9.999999999999997780e-01,
                              2.724275017472889693e-01, 7.361443106917049395e-01,
                              -7.385489458759963988e-01, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00],
                             [-2.724275017472889693e-01, 2.724275017472889693e-01,
                              1.000000000000000000e+00, -1.519200351467042132e-01,
                              1.952833664712358142e-01, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00],
                             [-7.361443106917048285e-01, 7.361443106917048285e-01,
                              -1.519200351467042132e-01, 1.000000000000000000e+00,
                              -8.751899489873674609e-01, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00],
                             [7.385489458759963988e-01, -7.385489458759963988e-01,
                              1.952833664712358142e-01, -8.751899489873673499e-01,
                              1.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00],
                             [0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 9.999999999999998890e-01,
                              0.000000000000000000e+00, 0.000000000000000000e+00],
                             [0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00,
                              1.000000000000000000e+00, 1.543033499620919125e-01],
                             [0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00,
                              0.000000000000000000e+00, 0.000000000000000000e+00,
                              1.543033499620919125e-01, 1.000000000000000000e+00]])

        assert hasattr(oz, 'ranks_')
        assert oz.ranks_.shape == (X.shape[1], X.shape[1])
        npt.assert_array_almost_equal(oz.ranks_, expected)

        # Image similarity comparision
        oz.finalize()
        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
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
        sys.platform == 'win32', reason="images not close on windows"
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


class TestRank1DQuick(VisualTestCase):
    """
    Test `rank1d` quick method
    """
    def test_rank1d_quick(self):
        """
        test rank2d using default parameters
        """
        X, y = load_energy()
        self.assert_images_similar(ax=rank1d(X, y))


class TestRank2DQuick(VisualTestCase):
    """
    Test `rank1d` quick method
    """
    def test_rank2d_quick(self):
        """
        Test rank1d using default parameters
        """
        X, y = load_energy()
        self.assert_images_similar(ax=rank2d(X, y))