# tests.test_regressor.test_residuals
# Ensure that the regressor residuals visualizations work.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  Sat Oct 8 16:30:39 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_residuals.py [7d3f5e6] benjamin@bengfort.com $

"""
Ensure that the regressor residuals visualizations work.
"""

##########################################################################
## Imports
##########################################################################

import pytest
import matplotlib.pyplot as plt

from yellowbrick.regressor.residuals import *

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin, Dataset, Split

from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split as tts


try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Data
##########################################################################

@pytest.fixture(scope='class')
def data(request):
    """
    Creates a fixture of train and test splits for the sklearn digits dataset
    For ease of use returns a Dataset named tuple composed of two Split tuples.
    """
    X, y = make_regression(
        n_samples=500, n_features=22, n_informative=8, random_state=42
    )

    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.2, random_state=11
    )

    # Set a class attribute for digits
    request.cls.data = Dataset(
        Split(X_train, X_test), Split(y_train, y_test)
    )


##########################################################################
## Prediction Error Test Cases
##########################################################################


@pytest.mark.usefixtures("data")
class TestPredictionError(VisualTestCase, DatasetMixin):
    """
    Test the PredictionError visualizer
    """

    def test_pred_error_integration(self):
        """
        Integration test with image similarity on random data with SVR
        """
        _, ax = plt.subplots()

        visualizer = PredictionError(SVR(), ax=ax)

        visualizer.fit(self.data.X.train, self.data.y.train)
        visualizer.score(self.data.X.test, self.data.y.test)
        visualizer.finalize()

        self.assert_images_similar(visualizer, tol=10)

    @pytest.mark.skipif(pd is None, reason="pandas is required")
    def test_pred_error_integration_pandas(self):
        """
        Test Pandas real world dataset with image similarity on Ridge
        """
        _, ax = plt.subplots()

        # Load the occupancy dataset from fixtures
        data = self.load_data('energy')
        target = 'cooling_load'
        features = [
            "relative_compactness", "surface_area", "wall_area", "roof_area",
            "overall_height", "orientation", "glazing_area",
            "glazing_area_distribution"
        ]

        # Create instances and target
        X = pd.DataFrame(data[features])
        y = pd.Series(data[target].astype(float))

        # Create train/test splits
        splits = tts(X, y, test_size=0.2, random_state=8873)
        X_train, X_test, y_train, y_test = splits

        visualizer = PredictionError(Ridge(), ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()

        self.assert_images_similar(visualizer, tol=10)

    def test_score(self):
        """
        Assert returns R2 score
        """
        visualizer = PredictionError(LinearRegression())

        visualizer.fit(self.data.X.train, self.data.y.train)
        score = visualizer.score(self.data.X.test, self.data.y.test)

        assert score == pytest.approx(1.0)
        assert visualizer.score_ == score

    @pytest.mark.skip(reason="not implemented yet")
    def test_peplot_shared_limits(self):
        """
        Test shared limits on the peplot
        """
        raise NotImplementedError("not yet implemented")

    @pytest.mark.skip(reason="not implemented yet")
    def test_peplot_draw_bounds(self):
        """
        Test the peplot +/- one bounding in draw
        """
        raise NotImplementedError("not yet implemented")


##########################################################################
## Residuals Plot Test Cases
##########################################################################

@pytest.mark.usefixtures("data")
class TestResidualsPlot(VisualTestCase, DatasetMixin):
    """
    Test ResidualPlot visualizer
    """

    def test_residuals_plot_integration(self):
        """
        Integration test with image similarity on random data with OLS
        """
        _, ax = plt.subplots()

        visualizer = ResidualsPlot(LinearRegression(), ax=ax)

        visualizer.fit(self.data.X.train, self.data.y.train)
        visualizer.score(self.data.X.test, self.data.y.test)
        visualizer.finalize()

        self.assert_images_similar(visualizer, tol=10)

    @pytest.mark.skipif(pd is None, reason="pandas is required")
    def test_residuals_plot_integration_pandas(self):
        """
        Test Pandas real world dataset with image similarity on Lasso
        """
        _, ax = plt.subplots()

        # Load the occupancy dataset from fixtures
        data = self.load_data('energy')
        target = 'heating_load'
        features = [
            "relative_compactness", "surface_area", "wall_area", "roof_area",
            "overall_height", "orientation", "glazing_area",
            "glazing_area_distribution"
        ]

        # Create instances and target
        X = pd.DataFrame(data[features])
        y = pd.Series(data[target].astype(float))

        # Create train/test splits
        splits = tts(X, y, test_size=0.2, random_state=231)
        X_train, X_test, y_train, y_test = splits

        visualizer = ResidualsPlot(Lasso(), ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()

        self.assert_images_similar(visualizer, tol=10)

    def test_score(self):
        """
        Assert returns R2 score
        """
        visualizer = ResidualsPlot(SVR())

        visualizer.fit(self.data.X.train, self.data.y.train)
        score = visualizer.score(self.data.X.test, self.data.y.test)

        assert score == pytest.approx(0.03344393985277794)
        assert visualizer.train_score_ == pytest.approx(0.04743502276335876)
        assert visualizer.test_score_ == score
