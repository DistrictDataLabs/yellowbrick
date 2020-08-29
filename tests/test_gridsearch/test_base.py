# # tests.test_gridsearch.test_base.py
# # Test the GridSearchColorPlot (standard and quick visualizers).
# #
# # Author:   Tan Tran
# # Created:  Sat Aug 29 12:00:00 2020 -0400
# #
# # Copyright (C) 2020 The scikit-yb developers
# # For license information, see LICENSE.txt
# #

"""
Test the GridSearchColorPlot visualizer.
"""

# ##########################################################################
# ## Imports
# ##########################################################################

import pytest

from tests.base import VisualTestCase
from tests.fixtures import Dataset

from yellowbrick.datasets import load_occupancy
from yellowbrick.gridsearch import GridSearchColorPlot, gridsearch_color_plot

from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import pandas as pd

# ##########################################################################
# ## Test fixtures
# ##########################################################################

@pytest.fixture(scope="class")
def binary(request):
    """
    Creates a random binary classification dataset fixture
    """
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=2,
        n_classes=2,
        n_clusters_per_class=2,
        random_state=1234,
    )

    request.cls.binary = Dataset(X, y)

@pytest.fixture(scope="class")
def gridsearchcv(request):
    """
    Creates an sklearn SVC, a GridSearchCV for testing through the SVC's kernel,
    gamma, and C parameters, and returns the GridSearchCV.
    """

    svc = SVC()
    grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.01, 0.1, 1, 10]},
            {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10]}]
    gridsearchcv = GridSearchCV(svc, grid, n_jobs=4)

    request.cls.gridsearchcv = gridsearchcv

@pytest.mark.usefixtures("binary", "gridsearchcv")
class TestGridSearchColorPlot(VisualTestCase):
    """
    Tests of basic GridSearchColorPlot functionality
    """

    # ##########################################################################
    # ## GridSearchColorPlot Base Test Cases
    # ##########################################################################

    def test_gridsearchcolorplot(self):
        """
        Test GridSearchColorPlot drawing
        """
        
        gs_viz = GridSearchColorPlot(self.gridsearchcv, 'C', 'kernel')
        gs_viz.fit(self.binary.X, self.binary.y)
        self.assert_images_similar(gs_viz)

    def test_quick_method(self):
        """
        Test gridsearch_color_plot quick method
        """

        gs = self.gridsearchcv

        # If no X data is passed to quick method, model is assumed to be fit
        # already 
        gs.fit(self.binary.X, self.binary.y)

        gs_viz = gridsearch_color_plot(gs, 'gamma', 'C')
        assert isinstance(gs_viz, GridSearchColorPlot)
        self.assert_images_similar(gs_viz)

    # ##########################################################################
    # ## Integration Tests
    # ##########################################################################

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_integration(self):
        """
        Test GridSearchColorPlot on sklearn occupancy data set (as pandas df)
        """
        
        X, y = load_occupancy(return_dataset=True).to_pandas()
        X, y = X.head(1000), y.head(1000)

        gs_viz = GridSearchColorPlot(self.gridsearchcv, 'C', 'kernel')
        gs_viz.fit(X, y)

        self.assert_images_similar(gs_viz)

    def test_numpy_integration(self):
        """
        Test GridSearchColorPlot on sklearn occupancy data set (as numpy df)
        """

        X, y = load_occupancy(return_dataset=True).to_numpy()
        X, y = X[:1000], y[:1000]

        gs_viz = GridSearchColorPlot(self.gridsearchcv, 'C', 'kernel')
        gs_viz.fit(X, y)

        self.assert_images_similar(gs_viz)