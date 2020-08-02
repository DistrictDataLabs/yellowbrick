# tests.test_regressor.test_prediction_error
# Ensure that the regressor prediction error visualization works.
#
# Author:   Rebecca Bilbro
# Author:   Benjamin Bengfort
# Created:  Sat Oct 8 16:30:39 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_prediction_error.py []  $

"""
Ensure that the regressor prediction error visualization works.
"""

##########################################################################
## Imports
##########################################################################

import pytest
import matplotlib.pyplot as plt

from unittest import mock
from tests.fixtures import Dataset, Split
from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase

from yellowbrick.datasets import load_energy
from yellowbrick.regressor.prediction_error import PredictionError, prediction_error

from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts

try:
    import pandas as pd
except ImportError:
    pd = None

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

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_prediction_error_quick_method(self):
        """
        Image similarity test using the residuals plot quick method
        """
        _, ax = plt.subplots()

        model = Lasso(random_state=19)
        oz = prediction_error(
            model, self.data.X.train, self.data.y.train, ax=ax, show=False
        )
        assert isinstance(oz, PredictionError)
        self.assert_images_similar(oz)
