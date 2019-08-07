# tests.test_regressor.test_residuals
# Ensure that the regressor residuals visualizations work.
#
# Author:   Rebecca Bilbro
# Author:   Benjamin Bengfort
# Created:  Sat Oct 8 16:30:39 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_residuals.py [7d3f5e6] benjamin@bengfort.com $

"""
Ensure that the regressor residuals visualizations work.
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt

from yellowbrick.datasets import load_energy
from yellowbrick.regressor.residuals import *
from yellowbrick.exceptions import YellowbrickValueError

from unittest import mock
from tests.fixtures import Dataset, Split
from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase

from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts


try:
    import pandas as pd
except ImportError:
    pd = None

# Determine version of matplotlib
MPL_VERS_MAJ = int(mpl.__version__.split(".")[0])


##########################################################################
## Data
##########################################################################


@pytest.fixture(scope="class")
def data(request):
    """
    Creates a fixture of train and test splits for the sklearn digits dataset
    For ease of use returns a Dataset named tuple composed of two Split tuples.
    """
    X, y = make_regression(
        n_samples=500,
        n_features=22,
        n_informative=8,
        random_state=42,
        noise=0.2,
        bias=0.2,
    )

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=11)

    # Set a class attribute for digits
    request.cls.data = Dataset(Split(X_train, X_test), Split(y_train, y_test))


##########################################################################
## Prediction Error Test Cases
##########################################################################


@pytest.mark.usefixtures("data")
class TestPredictionError(VisualTestCase):
    """
    Test the PredictionError visualizer
    """

    @pytest.mark.filterwarnings("ignore:Stochastic Optimizer")
    @pytest.mark.filterwarnings("ignore:internal gelsd driver lwork query error")
    def test_prediction_error(self):
        """
        Test image similarity of prediction error on random data
        """
        _, ax = plt.subplots()

        model = MLPRegressor(random_state=229)
        visualizer = PredictionError(model, ax=ax)

        visualizer.fit(self.data.X.train, self.data.y.train)
        visualizer.score(self.data.X.test, self.data.y.test)
        visualizer.finalize()

        self.assert_images_similar(visualizer, tol=1, remove_legend=True)

    @pytest.mark.skipif(pd is None, reason="pandas is required")
    def test_prediction_error_pandas(self):
        """
        Test Pandas real world dataset with image similarity on Ridge
        """
        _, ax = plt.subplots()

        # Load the occupancy dataset from fixtures
        data = load_energy(return_dataset=True)
        X, y = data.to_pandas()

        # Create train/test splits
        splits = tts(X, y, test_size=0.2, random_state=8873)
        X_train, X_test, y_train, y_test = splits

        visualizer = PredictionError(Ridge(random_state=22), ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()

        self.assert_images_similar(visualizer, tol=1, remove_legend=True)

    def test_prediction_error_numpy(self):
        """
        Test NumPy real world dataset with image similarity on Ridge
        """
        _, ax = plt.subplots()

        # Load the occupancy dataset from fixtures
        data = load_energy(return_dataset=True)
        X, y = data.to_numpy()

        # Create train/test splits
        splits = tts(X, y, test_size=0.2, random_state=8873)
        X_train, X_test, y_train, y_test = splits

        visualizer = PredictionError(Ridge(random_state=22), ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()

        self.assert_images_similar(visualizer, tol=1, remove_legend=True)

    def test_score(self):
        """
        Assert returns R2 score
        """
        visualizer = PredictionError(LinearRegression())

        visualizer.fit(self.data.X.train, self.data.y.train)
        score = visualizer.score(self.data.X.test, self.data.y.test)

        assert score == pytest.approx(0.9999983124154965)
        assert visualizer.score_ == score

    def test_peplot_shared_limits(self):
        """
        Test shared limits on the peplot
        """
        visualizer = PredictionError(LinearRegression(), shared_limits=False)

        visualizer.fit(self.data.X.train, self.data.y.train)
        visualizer.score(self.data.X.test, self.data.y.test)
        visualizer.finalize()

        xlim = tuple(map(int, visualizer.ax.get_xlim()))
        ylim = tuple(map(int, visualizer.ax.get_ylim()))
        assert xlim == ylim

    @pytest.mark.filterwarnings("ignore:internal gelsd driver lwork query error")
    def test_peplot_no_shared_limits(self):
        """
        Test image similarity with no shared limits on the peplot
        """
        visualizer = PredictionError(Ridge(random_state=43), shared_limits=False)

        visualizer.fit(self.data.X.train, self.data.y.train)
        visualizer.score(self.data.X.test, self.data.y.test)
        visualizer.finalize()

        xlim = tuple(map(int, visualizer.ax.get_xlim()))
        ylim = tuple(map(int, visualizer.ax.get_ylim()))
        assert not xlim == ylim

        self.assert_images_similar(visualizer, tol=1.0, remove_legend=True)

    def test_peplot_no_lines(self):
        """
        Test image similarity with no lines drawn on the plot
        """
        visualizer = PredictionError(
            Lasso(random_state=23, alpha=10), bestfit=False, identity=False
        )

        visualizer.fit(self.data.X.train, self.data.y.train)
        visualizer.score(self.data.X.test, self.data.y.test)
        visualizer.finalize()

        self.assert_images_similar(visualizer, tol=1.0, remove_legend=True)

    def test_alpha_param(self):
        """
        Test that the user can supply an alpha param on instantiation
        """
        # Instantiate a sklearn regressor
        model = Lasso(random_state=23, alpha=10)
        # Instantiate a prediction error plot, provide custom alpha
        visualizer = PredictionError(model, bestfit=False, identity=False, alpha=0.7)

        # Test param gets set correctly
        assert visualizer.alpha == 0.7

        # Mock ax and fit the visualizer
        visualizer.ax = mock.MagicMock(autospec=True)
        visualizer.fit(self.data.X.train, self.data.y.train)
        visualizer.score(self.data.X.test, self.data.y.test)

        # Test that alpha was passed to internal matplotlib scatterplot
        _, scatter_kwargs = visualizer.ax.scatter.call_args
        assert "alpha" in scatter_kwargs
        assert scatter_kwargs["alpha"] == 0.7

    @pytest.mark.xfail(
        reason="""third test fails with AssertionError: Expected fit
        to be called once. Called 0 times."""
    )
    def test_peplot_with_fitted(self):
        """
        Test that PredictionError properly handles an already-fitted model
        """
        X, y = load_energy(return_dataset=True).to_numpy()

        model = Ridge().fit(X, y)

        with mock.patch.object(model, "fit") as mockfit:
            oz = PredictionError(model)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with mock.patch.object(model, "fit") as mockfit:
            oz = PredictionError(model, is_fitted=True)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with mock.patch.object(model, "fit") as mockfit:
            oz = PredictionError(model, is_fitted=False)
            oz.fit(X, y)
            mockfit.assert_called_once_with(X, y)


##########################################################################
## Residuals Plot Test Cases
##########################################################################


@pytest.mark.usefixtures("data")
class TestResidualsPlot(VisualTestCase):
    """
    Test ResidualPlot visualizer
    """

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_residuals_plot(self):
        """
        Image similarity of residuals plot on random data with OLS
        """
        _, ax = plt.subplots()

        visualizer = ResidualsPlot(LinearRegression(), ax=ax)

        visualizer.fit(self.data.X.train, self.data.y.train)
        visualizer.score(self.data.X.test, self.data.y.test)
        visualizer.finalize()

        self.assert_images_similar(visualizer)

    @pytest.mark.xfail(
        sys.platform == "win32", reason="images not close on windows (RMSE=32)"
    )
    @pytest.mark.filterwarnings("ignore:Stochastic Optimizer")
    def test_residuals_plot_no_histogram(self):
        """
        Image similarity test when hist=False
        """
        _, ax = plt.subplots()

        model = MLPRegressor(random_state=19)
        visualizer = ResidualsPlot(model, ax=ax, hist=False)

        visualizer.fit(self.data.X.train, self.data.y.train)
        visualizer.score(self.data.X.test, self.data.y.test)
        visualizer.finalize()

        self.assert_images_similar(visualizer, tol=1, remove_legend=True)

    @pytest.mark.skipif(
        MPL_VERS_MAJ >= 2, reason="test requires mpl earlier than 2.0.2"
    )
    def test_hist_matplotlib_version(self, mock_toolkit):
        """
        ValueError is raised when matplotlib version is incorrect and hist=True
        """
        with pytst.raises(ImportError):
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            assert not make_axes_locatable

        with pytest.raises(YellowbrickValueError, match="requires matplotlib 2.0.2"):
            ResidualsPlot(LinearRegression(), hist=True)

    @pytest.mark.skipif(
        MPL_VERS_MAJ >= 2, reason="test requires mpl earlier than 2.0.2"
    )
    def test_no_hist_matplotlib_version(self, mock_toolkit):
        """
        No error is raised when matplotlib version is incorrect and hist=False
        """
        with pytest.raises(ImportError):
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            assert not make_axes_locatable

        try:
            ResidualsPlot(LinearRegression(), hist=False)
        except YellowbrickValueError as e:
            self.fail(e)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_residuals_quick_method(self):
        """
        Image similarity test using the residuals plot quick method
        """
        _, ax = plt.subplots()

        model = Lasso(random_state=19)
        oz = residuals_plot(
            model, self.data.X.train, self.data.y.train, ax=ax, random_state=23
        )

        self.assert_images_similar(oz)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    @pytest.mark.skipif(pd is None, reason="pandas is required")
    def test_residuals_plot_pandas(self):
        """
        Test Pandas real world dataset with image similarity on Lasso
        """
        _, ax = plt.subplots()

        # Load the occupancy dataset from fixtures
        data = load_energy(return_dataset=True)
        X, y = data.to_pandas()

        # Create train/test splits
        splits = tts(X, y, test_size=0.2, random_state=231)
        X_train, X_test, y_train, y_test = splits

        visualizer = ResidualsPlot(Lasso(random_state=44), ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()

        self.assert_images_similar(visualizer)

    def test_residuals_plot_numpy(self):
        """
        Test NumPy real world dataset with image similarity on Lasso
        """
        _, ax = plt.subplots()

        # Load the occupancy dataset from fixtures
        data = load_energy(return_dataset=True)
        X, y = data.to_numpy()

        # Create train/test splits
        splits = tts(X, y, test_size=0.2, random_state=231)
        X_train, X_test, y_train, y_test = splits

        visualizer = ResidualsPlot(Lasso(random_state=44), ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()

        self.assert_images_similar(visualizer, tol=1.5)

    def test_score(self):
        """
        Assert returns R2 score
        """
        visualizer = ResidualsPlot(Ridge(random_state=8893))

        visualizer.fit(self.data.X.train, self.data.y.train)
        score = visualizer.score(self.data.X.test, self.data.y.test)

        assert score == pytest.approx(0.9999888484, rel=1e-4)
        assert visualizer.train_score_ == pytest.approx(0.9999906, rel=1e-4)
        assert visualizer.test_score_ == score

    @mock.patch("yellowbrick.regressor.residuals.plt.sca", autospec=True)
    def test_alpha_param(self, mock_sca):
        """
        Test that the user can supply an alpha param on instantiation
        """
        # Instantiate a prediction error plot, provide custom alpha
        visualizer = ResidualsPlot(
            Ridge(random_state=8893), train_alpha=0.3, test_alpha=0.75, hist=False
        )
        alphas = {"train_point": 0.3, "test_point": 0.75}
        # Test param gets set correctly
        assert visualizer.alphas == alphas

        visualizer.ax = mock.MagicMock()
        visualizer.fit(self.data.X.train, self.data.y.train)
        visualizer.score(self.data.X.test, self.data.y.test)

        # Test that alpha was passed to internal matplotlib scatterplot
        _, scatter_kwargs = visualizer.ax.scatter.call_args
        assert "alpha" in scatter_kwargs
        assert scatter_kwargs["alpha"] == 0.75

    @pytest.mark.xfail(
        reason="""third test fails with AssertionError: Expected fit
        to be called once. Called 0 times."""
    )
    def test_residuals_with_fitted(self):
        """
        Test that ResidualsPlot properly handles an already-fitted model
        """
        X, y = load_energy(return_dataset=True).to_numpy()

        model = Ridge().fit(X, y)

        with mock.patch.object(model, "fit") as mockfit:
            oz = ResidualsPlot(model)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with mock.patch.object(model, "fit") as mockfit:
            oz = ResidualsPlot(model, is_fitted=True)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with mock.patch.object(model, "fit") as mockfit:
            oz = ResidualsPlot(model, is_fitted=False)
            oz.fit(X, y)
            mockfit.assert_called_once_with(X, y)
