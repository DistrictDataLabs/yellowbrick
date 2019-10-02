# tests.test_contrib.test_scatter
# Test the ScatterViz feature analysis visualizers
#
# Author:   Nathan Danielsen
# Created:  Fri Feb 26 19:40:00 2017 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_scatter.py [a89633e] benjamin@bengfort.com $
"""
Test the ScatterViz feature analysis visualizers
"""

##########################################################################
# Imports
##########################################################################

import pytest
import numpy as np

from unittest import mock
from tests.base import VisualTestCase
from yellowbrick.contrib.scatter import *
from yellowbrick.datasets import load_occupancy
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.style import palettes

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
# ScatterViz Base Tests
##########################################################################


@pytest.mark.filterwarnings("ignore")
class TestScatterViz(VisualTestCase):
    """
    Test ScatterViz
    """

    # yapf: disable
    X = np.array([
        [2.318, 2.727, 4.260, 7.212, 4.792],
        [2.315, 2.726, 4.295, 7.140, 4.783, ],
        [2.315, 2.724, 4.260, 7.135, 4.779, ],
        [2.110, 3.609, 4.330, 7.985, 5.595, ],
        [2.110, 3.626, 4.330, 8.203, 5.621, ],
        [2.110, 3.620, 4.470, 8.210, 5.612, ]
    ])
    # yapf: enable
    y = np.array([1, 0, 1, 0, 1, 0])

    def test_init_alias(self):
        """
        Test alias for ScatterViz
        """
        features = ["temperature", "relative humidity"]
        visualizer = ScatterVisualizer(features=features, markers=["*"])
        assert visualizer.markers is not None

    def test_scatter(self):
        """
        Assert no errors occur during scatter visualizer integration
        """
        X_two_cols = self.X[:, :2]
        features = ["temperature", "relative humidity"]
        visualizer = ScatterViz(features=features)
        visualizer.fit_transform(X_two_cols, self.y)

    def test_color_builds(self):
        """
        Assert no errors occur during scatter visualizer integration
        """
        colors = palettes.PALETTES["pastel"]
        X_two_cols = self.X[:, :2]
        features = ["temperature", "relative humidity"]
        visualizer = ScatterViz(features=features, color=colors)
        visualizer.fit_transform(X_two_cols, self.y)

    def test_scatter_no_features(self):
        """
        Assert no errors during scatter visualizer integration - no features
        """
        X_two_cols = self.X[:, :2]
        visualizer = ScatterViz()
        visualizer.fit_transform_show(X_two_cols, self.y)
        assert visualizer.features_ == ["Feature One", "Feature Two"]

    def test_scatter_only_two_features_allowed_init(self):
        """
        Assert that only two features are allowed for scatter visualizer init
        """
        features = ["temperature", "relative humidity", "light"]

        with pytest.raises(YellowbrickValueError):
            ScatterViz(features=features)

    def test_scatter_xy_and_features_raise_error(self):
        """
        Assert that x,y and features will raise scatterviz error
        """
        features = ["temperature", "relative humidity", "light"]

        with pytest.raises(YellowbrickValueError):
            ScatterViz(features=features, x="one", y="two")

    def test_scatter_xy_changes_to_features(self):
        """
        Assert that x,y with no features will not raise scatterviz error
        """
        visualizer = ScatterViz(x="one", y="two")
        assert visualizer.features == ["one", "two"]

    def test_scatter_requires_two_features_in_numpy_matrix(self):
        """
        Assert only two features allowed for scatter visualizer if not in init
        """
        visualizer = ScatterViz()
        with pytest.raises(YellowbrickValueError, match="only accepts two features"):
            visualizer.fit_transform(self.X, self.y)

    def test_integrated_scatter(self):
        """
        Test scatter on the real, occupancy data set
        """
        # Load the data from the fixture
        X, y = load_occupancy(return_dataset=True).to_numpy()

        # Test the visualizer
        features = ["temperature", "relative humidity"]
        visualizer = ScatterViz(features=features)
        visualizer.fit_transform_show(X[:, :2], y)

    def test_alpha_param(self):
        """
        Test that the user can supply an alpha param on instantiation
        """
        # Instantiate a scatter plot and provide a custom alpha
        visualizer = ScatterVisualizer(alpha=0.7, features=["a", "b"])

        # Test param gets set correctly
        assert visualizer.alpha == 0.7

        # Mock ax and fit the visualizer
        visualizer.ax = mock.MagicMock(autospec=True)
        visualizer.fit(self.X[:, :2], self.y)

        # Test that alpha was passed to the scatter plot
        _, scatter_kwargs = visualizer.ax.scatter.call_args
        assert "alpha" in scatter_kwargs
        assert scatter_kwargs["alpha"] == 0.7

    def test_scatter_quick_method(self):
        """
        Test scatter quick method on the real, occupancy data set
        """
        # Load the data from the fixture
        X, y = load_occupancy(return_dataset=True).to_numpy()

        # Test the visualizer
        features = ["temperature", "relative humidity"]
        viz = scatterviz(X[:, :2], y=y, ax=None, features=features)

        # test that is returns a matplotlib obj with axes
        assert isinstance(viz, ScatterVisualizer)

    @pytest.mark.skipif(pd is None, reason="pandas is required for this test")
    def test_integrated_scatter_with_pandas(self):
        """
        Test scatterviz on the real, occupancy data set with pandas
        """
        # Load the data from the fixture
        # Load the data from the fixture
        X, y = load_occupancy(return_dataset=True).to_pandas()

        # Test the visualizer
        features = ["temperature", "relative humidity"]
        visualizer = ScatterViz(features=features)
        visualizer.fit_transform_show(X, y)

    @pytest.mark.xfail(reason="numpy structured arrays have changed since v1.14")
    def test_integrated_scatter_numpy_named_arrays(self):
        """
        Test scatterviz on numpy named arrays
        """
        dt = np.dtype(
            {
                "names": ["one", "two", "three", "four", "five"],
                "formats": [np.float64, np.float64, np.float64, np.float64, np.float64],
            }
        )

        X_named = self.X.astype(dt, casting="unsafe")
        visualizer = ScatterViz(features=["one", "two"])
        visualizer.fit_transform_show(X_named, self.y)
        assert visualizer.features_ == ["one", "two"]

    def test_integrated_scatter_numpy_arrays_no_names(self):
        """
        Test scaterviz on regular numpy arrays
        """
        visualizer = ScatterViz(features=[1, 2])
        visualizer.fit_transform_show(self.X, self.y)
        assert visualizer.features_ == [1, 2]

    def test_scatter_image(self):
        """
        Test the scatterviz image similarity
        """
        # self.setUp_ImageTest()

        X_two_cols = self.X[:, :2]
        features = ["temperature", "relative humidity"]
        visualizer = ScatterViz(features=features)
        visualizer.fit(X_two_cols, self.y)
        visualizer.draw(X_two_cols, self.y)

        self.assert_images_similar(visualizer)
