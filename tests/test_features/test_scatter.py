# tests.test_features.test_scatter
# Test the ScatterViz feature analysis visualizers
#
# Author:   Nathan Danielsen <nathan.danielsen@gmail.com>
# Created:  Fri Feb 26 19:40:00 2017 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_scatter.py [fc94ec4] ndanielsen@users.noreply.github.com $
"""
Test the ScatterViz feature analysis visualizers
"""

##########################################################################
# Imports
##########################################################################

import unittest
import numpy as np
import matplotlib as mptl

from yellowbrick.features.scatter import *
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.style import palettes

from tests.dataset import DatasetMixin
from tests.base import VisualTestCase
from matplotlib.testing.exceptions import ImageComparisonFailure

try:
    import pandas
except ImportError:
    pandas = None

##########################################################################
# ScatterViz Base Tests
##########################################################################


class ScatterVizTests(VisualTestCase, DatasetMixin):

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

    def setUp(self):
        self.occupancy = self.load_data('occupancy')
        super(ScatterVizTests, self).setUp()

    def tearDown(self):
        self.occupancy = None
        super(ScatterVizTests, self).tearDown()

    def test_init_alias(self):
        features = ["temperature", "relative_humidity"]
        visualizer = ScatterVisualizer(features=features, markers=['*'])
        self.assertIsNotNone(visualizer.markers)

    def test_scatter(self):
        """
        Assert no errors occur during scatter visualizer integration
        """
        X_two_cols = self.X[:, :2]
        features = ["temperature", "relative_humidity"]
        visualizer = ScatterViz(features=features)
        visualizer.fit_transform(X_two_cols, self.y)

    def test_color_builds(self):
        """
        Assert no errors occur during scatter visualizer integration
        """
        colors = palettes.PALETTES['pastel']
        X_two_cols = self.X[:, :2]
        features = ["temperature", "relative_humidity"]
        visualizer = ScatterViz(features=features, color=colors)
        visualizer.fit_transform(X_two_cols, self.y)

    def test_scatter_no_features(self):
        """
        Assert no errors during scatter visualizer integration - no features
        """
        X_two_cols = self.X[:, :2]
        visualizer = ScatterViz()
        visualizer.fit_transform_poof(X_two_cols, self.y)
        self.assertEquals(visualizer.features_, ['Feature One', 'Feature Two'])

    def test_scatter_only_two_features_allowed_init(self):
        """
        Assert that only two features are allowed for scatter visualizer init
        """
        features = ["temperature", "relative_humidity", "light"]

        with self.assertRaises(YellowbrickValueError):
            ScatterViz(features=features)

    def test_scatter_xy_and_features_raise_error(self):
        """
        Assert that x,y and features will raise scatterviz error
        """
        features = ["temperature", "relative_humidity", "light"]

        with self.assertRaises(YellowbrickValueError):
            ScatterViz(features=features, x='one', y='two')

    def test_scatter_xy_changes_to_features(self):
        """
        Assert that x,y with no features will not raise scatterviz error
        """
        visualizer = ScatterViz(x='one', y='two')
        self.assertEquals(visualizer.features_, ['one', 'two'])

    def test_scatter_requires_two_features_in_numpy_matrix(self):
        """
        Assert only two features allowed for scatter visualizer if not in init
        """
        visualizer = ScatterViz()
        with self.assertRaises(YellowbrickValueError) as context:
            visualizer.fit_transform(self.X, self.y)
            self.assertTrue(
                'only accepts two features' in str(context.exception))

    def test_integrated_scatter(self):
        """
        Test scatter on the real, occupancy data set
        """
        # Load the data from the fixture
        X = self.occupancy[[
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]]

        # Convert to numpy arrays
        X = X.copy().view((float, len(X.dtype.names)))
        y = self.occupancy['occupancy'].astype(int)

        # Test the visualizer
        features = ["temperature", "relative_humidity"]
        visualizer = ScatterViz(features=features)
        visualizer.fit_transform_poof(X[:, :2], y)

    def test_scatter_quick_method(self):
        """
        Test scatter quick method on the real, occupancy data set
        """
        # Load the data from the fixture
        X = self.occupancy[[
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]]

        # Convert to numpy arrays
        X = X.copy().view((float, len(X.dtype.names)))
        y = self.occupancy['occupancy'].astype(int)

        # Test the visualizer
        features = ["temperature", "relative_humidity"]
        ax = scatterviz(X[:, :2], y=y, ax=None, features=features)

        # test that is returns a matplotlib obj with axes
        self.assertIsInstance(ax, mptl.axes.Axes)

    @unittest.skipUnless(pandas is not None,
                         "Pandas is not installed, could not run test.")
    def test_integrated_scatter_with_pandas(self):
        """
        Test scatterviz on the real, occupancy data set with pandas
        """
        # Load the data from the fixture
        X = self.occupancy[[
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]]
        y = self.occupancy['occupancy'].astype(int)

        # Convert X to a pandas dataframe
        X = pandas.DataFrame(X)
        X.columns = [
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]

        # Test the visualizer
        features = ["temperature", "relative_humidity"]
        visualizer = ScatterViz(features=features)
        visualizer.fit_transform_poof(X, y)

    def test_integrated_scatter_numpy_named_arrays(self):
        """
        Test scatterviz on numpy named arrays
        """
        dt = np.dtype({
            'names': ['one', 'two', 'three', 'four', "five"],
            'formats': [
                np.float64,
                np.float64,
                np.float64,
                np.float64,
                np.float64,
            ]
        })

        X_named = self.X.astype(dt, casting='unsafe')
        visualizer = ScatterViz(features=['one', 'two'])
        visualizer.fit_transform_poof(X_named, self.y)
        self.assertEquals(visualizer.features_, ['one', 'two'])


    def test_integrated_scatter_numpy_arrays_no_names(self):
        """
        Test scaterviz on regular numpy arrays
        """
        visualizer = ScatterViz(features=[1, 2])
        visualizer.fit_transform_poof(self.X, self.y)
        self.assertEquals(visualizer.features_, [1, 2])

    def test_scatter_image(self):
        """
        Test the scatterviz image similarity
        """
        # self.setUp_ImageTest()

        X_two_cols = self.X[:, :2]
        features = ["temperature", "relative_humidity"]
        visualizer = ScatterViz(features=features)
        visualizer.fit(X_two_cols, self.y)
        visualizer.draw(X_two_cols, self.y)

        self.assert_images_similar(visualizer)


    def test_scatter_image_fail(self):
        """
        Assert bad image similarity on scatterviz errors
        """

        X_two_cols = self.X[:, :2]
        features = ["temperature", "relative_humidity"]
        visualizer = ScatterViz(features=features)
        visualizer.fit(X_two_cols, self.y)
        visualizer.draw(X_two_cols, self.y)

        with self.assertRaises(ImageComparisonFailure):
            self.assert_images_similar(visualizer)
