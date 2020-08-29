# tests.test_features.test_jointplot
# Test the JointPlot Visualizer
#
# Author:   Prema Damodaran Roman
# Created:  Mon Apr 10 21:00:54 2017 -0400
#
# Copyright (C) 2017 The scikit-yb developers.
# For license information, see LICENSE.txt
#
# ID: test_jointplot.py [9e008b0] pdamodaran@users.noreply.github.com $

"""
Test joint plot visualization methods.

These tests work differently depending on what version of matplotlib is
installed. If version 2.0.2 or greater is installed, then most tests will
execute, otherwise the histogram tests will skip and only the warning will
be tested.
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest
import numpy as np

from functools import partial
from unittest.mock import patch, MagicMock

from sklearn.datasets import make_classification, make_regression

from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.features.jointplot import *
from ..fixtures import Dataset

try:
    # Only available in Matplotlib >= 2.0.2
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    make_axes_locatable = None

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Fixtures
##########################################################################

# Random numpy array generators
rand1d = partial(np.random.rand, 120)
rand2col = partial(np.random.rand, 120, 2)
rand3col = partial(np.random.rand, 120, 3)


@pytest.fixture(scope="class")
def discrete(request):
    """
    Creates a simple 2-column dataset with a discrete target.
    """
    X, y = make_classification(
        n_samples=120,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=2221,
    )

    request.cls.discrete = Dataset(X, y)


@pytest.fixture(scope="class")
def continuous(request):
    """
    Creates a simple 2-column dataset with a continuous target.
    """
    X, y = make_regression(n_samples=120, n_features=2, random_state=1112)

    request.cls.continuous = Dataset(X, y)


##########################################################################
## JointPlot Tests
##########################################################################


@pytest.mark.usefixtures("discrete", "continuous")
class TestJointPlotNoHistogram(VisualTestCase):
    """
    Test the JointPlot visualizer without histograms
    """

    def test_invalid_columns_values(self):
        """
        Assert invalid columns arguments raise exception
        """
        with pytest.raises(YellowbrickValueError, match="invalid for joint plot"):
            JointPlot(columns=["a", "b", "c"], hist=False)

    def test_invalid_correlation_values(self):
        """
        Assert invalid correlation arguments raise an exception
        """
        with pytest.raises(YellowbrickValueError, match="invalid correlation method"):
            JointPlot(correlation="foo", hist=False)

    def test_invalid_kind_values(self):
        """
        Assert invalid kind arguments raise exception
        """
        for bad_kind in ("foo", None, 123):
            with pytest.raises(YellowbrickValueError, match="invalid joint plot kind"):
                JointPlot(kind=bad_kind, hist=False)

    def test_invalid_hist_values(self):
        """
        Assert invalid hist arguments raise exception
        """
        for bad_hist in ("foo", 123):
            with pytest.raises(
                YellowbrickValueError, match="invalid argument for hist"
            ):
                JointPlot(hist=bad_hist)

    def test_no_haxes(self):
        """
        Test that xhax and yhax are not available
        """
        oz = JointPlot(hist=False)
        with pytest.raises(AttributeError, match="histogram for the X axis"):
            oz.xhax

        with pytest.raises(AttributeError, match="histogram for the Y axis"):
            oz.yhax

    @patch("yellowbrick.features.jointplot.plt")
    def test_correlation(self, mplt):
        """
        Test correlation is correctly computed
        """
        x = self.discrete.X[:, 0]
        y = self.discrete.X[:, 1]

        cases = (
            ("pearson", -0.3847799883805261),
            ("spearman", -0.37301201472324463),
            ("covariance", -0.5535440619953924),
            ("kendalltau", -0.2504201680672269),
        )

        for alg, expected in cases:
            oz = JointPlot(hist=False, correlation=alg, columns=None)
            oz.ax = MagicMock()
            oz.fit(x, y)

            assert hasattr(oz, "corr_")
            assert oz.corr_ == pytest.approx(
                expected
            ), "{} not computed correctly".format(alg)

    def test_columns_none_invalid_x(self):
        """
        When self.columns=None validate X and y
        """
        bad_kws = (
            {"X": rand1d(), "y": None},
            {"X": rand3col(), "y": None},
            {"X": rand2col(), "y": rand1d()},
            {"X": rand3col(), "y": rand1d()},
            {"X": rand1d(), "y": rand2col()},
        )

        for kws in bad_kws:
            oz = JointPlot(columns=None, hist=False)
            with pytest.raises(
                YellowbrickValueError, match="when self.columns is None"
            ):
                oz.fit(**kws)

    def test_columns_none_x_y(self):
        """
        When self.columns=None image similarity with valid X and y
        """
        oz = JointPlot(hist=False, columns=None)
        assert oz.fit(self.discrete.X[:, 0], self.discrete.y) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        # Appveyor and Linux conda fail due to non-text-based differences
        self.assert_images_similar(oz, tol=2.5)

    def test_columns_none_x(self):
        """
        When self.columns=None image similarity with valid X, no y
        """
        oz = JointPlot(hist=False, columns=None)
        assert oz.fit(self.discrete.X) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        tol = (
            4.0 if sys.platform == "win32" else 0.01
        )  # Fails on AppVeyor with RMS 3.941
        self.assert_images_similar(oz, tol=tol)

    def test_columns_single_index_no_y(self):
        """
        When self.columns=int or str y must not be None
        """
        oz = JointPlot(columns="foo", hist=False)
        with pytest.raises(YellowbrickValueError, match="y must be specified"):
            oz.fit(rand2col(), y=None)

    def test_columns_single_invalid_index_numpy(self):
        """
        When self.columns=int validate the index in X
        """
        oz = JointPlot(columns=2, hist=False)
        with pytest.raises(IndexError, match="could not index column '2' into type"):
            oz.fit(self.continuous.X, self.continuous.y)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_columns_single_invalid_index_pandas(self):
        """
        When self.columns=str validate the index in X
        """
        oz = JointPlot(columns="foo", hist=False)
        X = pd.DataFrame(self.continuous.X, columns=["a", "b"])
        y = pd.Series(self.continuous.y)

        with pytest.raises(IndexError, match="could not index column 'foo' into type"):
            oz.fit(X, y)

    def test_columns_single_int_index_numpy(self):
        """
        When self.columns=int image similarity on numpy dataset
        """
        oz = JointPlot(columns=1, hist=False)
        assert oz.fit(self.continuous.X, self.continuous.y) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        # Appveyor and Linux conda failed based on non-text-based differences
        self.assert_images_similar(oz, tol=5)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_columns_single_str_index_pandas(self):
        """
        When self.columns=str image similarity on pandas dataset
        """
        oz = JointPlot(columns="a", hist=False)
        X = pd.DataFrame(self.continuous.X, columns=["a", "b"])
        y = pd.Series(self.continuous.y)
        assert oz.fit(X, y) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        # Appveyor and Linux conda failed based on non-text-based differences
        self.assert_images_similar(oz, tol=5.5)

    def test_columns_double_int_index_numpy_no_y(self):
        """
        When self.columns=[int, int] image similarity on numpy dataset no y
        """
        oz = JointPlot(columns=[0, 1], hist=False)
        assert oz.fit(self.discrete.X, y=None) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        tol = (
            4.0 if sys.platform == "win32" else 0.01
        )  # Fails on AppVeyor with RMS 3.941
        self.assert_images_similar(oz, tol=tol)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_columns_double_str_index_pandas_no_y(self):
        """
        When self.columns=[str, str] image similarity on pandas dataset no y
        """
        oz = JointPlot(columns=["a", "b"], hist=False)
        X = pd.DataFrame(self.continuous.X, columns=["a", "b"])
        assert oz.fit(X, y=None) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        tol = (
            4.0 if sys.platform == "win32" else 0.01
        )  # Fails on AppVeyor with RMS 3.911
        self.assert_images_similar(oz, tol=tol)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_columns_double_index_discrete_y(self):
        """
        When self.columns=[str, str] on DataFrame with discrete y
        """
        oz = JointPlot(columns=["a", "b"], hist=False)
        X = pd.DataFrame(self.discrete.X, columns=["a", "b"])
        y = pd.Series(self.discrete.y)
        assert oz.fit(X, y) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        tol = (
            4.0 if sys.platform == "win32" else 0.01
        )  # Fails on AppVeyor with RMS 3.940
        self.assert_images_similar(oz, tol=tol)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_columns_double_index_continuous_y(self):
        """
        When self.columns=[str, str] on DataFrame with continuous y
        """
        oz = JointPlot(columns=["a", "b"], hist=False)
        X = pd.DataFrame(self.continuous.X, columns=["a", "b"])
        y = pd.Series(self.continuous.y)
        assert oz.fit(X, y) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        tol = (
            4.0 if sys.platform == "win32" else 0.01
        )  # Fails on AppVeyor with RMS 3.911
        self.assert_images_similar(oz, tol=tol)


@pytest.mark.skipif(
    make_axes_locatable is not None, reason="requires matplotlib <= 2.0.1"
)
def test_matplotlib_version_error():
    """
    Assert an exception is raised with incompatible matplotlib versions
    """
    with pytest.raises(YellowbrickValueError):
        JointPlot(hist=True)


@patch("yellowbrick.features.jointplot.make_axes_locatable", None)
def test_matplotlib_incompatibility():
    """
    Assert an exception is raised if make_axes_locatable is None
    """
    with pytest.raises(YellowbrickValueError):
        JointPlot(hist=True)


@pytest.mark.usefixtures("discrete", "continuous")
@pytest.mark.skipif(make_axes_locatable is None, reason="requires matplotlib >= 2.0.2")
class TestJointPlotHistogram(VisualTestCase):
    """
    Test the JointPlot visualizer with histograms
    """

    def test_haxes_available(self):
        """
        Test that xhax and yhax are available
        """
        oz = JointPlot(hist=True)
        assert oz.xhax is not None
        assert oz.yhax is not None

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_columns_none_x_y_hist(self):
        """
        When self.columns=None image similarity with valid X and y
        """
        oz = JointPlot(hist=True, columns=None)
        assert oz.fit(self.discrete.X[:, 0], self.discrete.y) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        self.assert_images_similar(oz)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_columns_none_x_hist(self):
        """
        When self.columns=None image similarity with valid X, no y
        """
        oz = JointPlot(hist=True, columns=None)
        assert oz.fit(self.discrete.X) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        self.assert_images_similar(oz)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_columns_single_int_index_numpy_hist(self):
        """
        When self.columns=int image similarity on numpy dataset
        """
        oz = JointPlot(columns=1, hist=True)
        assert oz.fit(self.continuous.X, self.continuous.y) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        # Python 3.6 Travis ImageComparisonFailure: images not close (RMS 0.837)
        self.assert_images_similar(oz, tol=1.0)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_columns_single_str_index_pandas_hist(self):
        """
        When self.columns=str image similarity on pandas dataset
        """
        oz = JointPlot(columns="a", hist=True)
        X = pd.DataFrame(self.continuous.X, columns=["a", "b"])
        y = pd.Series(self.continuous.y)
        assert oz.fit(X, y) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        # Python 3.6 Travis ImageComparisonFailure: images not close (RMS 1.109)
        self.assert_images_similar(oz, tol=1.5)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_columns_double_int_index_numpy_no_y_hist(self):
        """
        When self.columns=[int, int] image similarity on numpy dataset no y
        """
        oz = JointPlot(columns=[0, 1], hist=True)
        assert oz.fit(self.discrete.X, y=None) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        self.assert_images_similar(oz)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_columns_double_str_index_pandas_no_y_hist(self):
        """
        When self.columns=[str, str] image similarity on pandas dataset no y
        """
        oz = JointPlot(columns=["a", "b"], hist=True)
        X = pd.DataFrame(self.continuous.X, columns=["a", "b"])
        assert oz.fit(X, y=None) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        self.assert_images_similar(oz)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_columns_double_index_discrete_y_hist(self):
        """
        When self.columns=[str, str] on DataFrame with discrete y
        """
        oz = JointPlot(columns=["a", "b"], hist=True)
        X = pd.DataFrame(self.discrete.X, columns=["a", "b"])
        y = pd.Series(self.discrete.y)
        assert oz.fit(X, y) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        self.assert_images_similar(oz)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_columns_double_index_continuous_y_hist(self):
        """
        When self.columns=[str, str] on DataFrame with continuous y
        """
        oz = JointPlot(columns=["a", "b"], hist=True)
        X = pd.DataFrame(self.continuous.X, columns=["a", "b"])
        y = pd.Series(self.continuous.y)
        assert oz.fit(X, y) is oz
        assert hasattr(oz, "corr_")

        oz.finalize()
        # Appveyor and Linux conda failed due to non-text-based differences
        self.assert_images_similar(oz, tol=4.0)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_quick_method(self):
        """
        Test the joint_plot quick method
        """
        oz = joint_plot(self.continuous.X, self.continuous.y, columns=1, show=False)
        assert isinstance(oz, JointPlot)
        assert hasattr(oz, "corr_")

        # Python 3.6 Travis ImageComparisonFailure: images not close (RMS 0.837)
        self.assert_images_similar(oz, tol=1.0)
