# tests.test_features.test_scatter
# Test the ScatterViz feature analysis visualizers
#
# Author:   Nathan Danielsen <nathan.danielsen@gmail.com>
# Created:  Fri Feb 26 19:40:00 2017 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_scatter.py [] nathan.danielsen@gmail.com $

"""
Test the ScatterViz feature analysis visualizers
"""

##########################################################################
## Imports
##########################################################################

import unittest
import numpy as np
import numpy.testing as npt
import matplotlib as mptl

from tests.dataset import DatasetMixin
from yellowbrick.features.scatter import *
from yellowbrick.exceptions import YellowbrickValueError


try:
    import pandas
except ImportError:
    pandas = None


##########################################################################
## ScatterViz Base Tests
##########################################################################

class ScatterVizTests(unittest.TestCase, DatasetMixin):

    X = np.array(
            [[ 2.318, 2.727, 4.260, 7.212, 4.792],
             [ 2.315, 2.726, 4.295, 7.140, 4.783,],
             [ 2.315, 2.724, 4.260, 7.135, 4.779,],
             [ 2.110, 3.609, 4.330, 7.985, 5.595,],
             [ 2.110, 3.626, 4.330, 8.203, 5.621,],
             [ 2.110, 3.620, 4.470, 8.210, 5.612,]]
        )

    y = np.array([1, 1, 0, 1, 0, 0])

    def setUp(self):
        self.occupancy = self.load_data('occupancy')

    def tearDown(self):
        self.occupancy = None

    def test_init_alias(self):
        features = ["temperature", "relative_humidity"]
        visualizer = ScatterVisualizer(features=features, markers=['*'])
        self.assertIsNotNone(visualizer.markers)

    def test_init_alias(self):
        features = ["temperature", "relative_humidity"]
        visualizer = ScatterViz(features=features, markers=['*'])
        self.assertIsNotNone(visualizer.markers)

    def test_scatter(self):
        """
        Assert no errors occur during scatter visualizer integration
        """
        X_two_cols = self.X[:,:2]
        features = ["temperature", "relative_humidity"]
        visualizer = ScatterViz(features=features)
        visualizer.fit_transform(X_two_cols, self.y)

    def test_scatter_only_two_features_allowed_init(self):
        """
        Assert that only two features are allowed for this visualizer in init
        """
        features = ["temperature", "relative_humidity", "light"]

        with self.assertRaises(YellowbrickValueError) as context:
            visualizer = ScatterViz(features=features)

    def test_scatter_requires_two_features_in_numpy_matrix(self):
        """
        Assert that only two features are allowed for this visualizer if not
        set in the init
        """
        visualizer = ScatterViz()
        with self.assertRaises(YellowbrickValueError) as context:
            visualizer.fit_transform(self.X, self.y)
            self.assertTrue('only accepts two features' in str(context.exception))


    def test_integrated_scatter(self):
        """
        Test scatter on the real, occupancy data set
        """
        # Load the data from the fixture
        X = self.occupancy[[
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]]
        y = self.occupancy['occupancy'].astype(int)

        # Convert X to an ndarray
        X = X.view((float, len(X.dtype.names)))

        # Test the visualizer
        features = ["temperature", "relative_humidity"]
        visualizer = ScatterViz(features=features)
        visualizer.fit_transform_poof(X[:,:2], y)


    def test_scatter_quick_method(self):
        """
        Test scatter on the real, occupancy data set
        """
        # Load the data from the fixture
        X = self.occupancy[[
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]]
        y = self.occupancy['occupancy'].astype(int)

        # Convert X to an ndarray
        X = X.view((float, len(X.dtype.names)))

        # Test the visualizer
        features = ["temperature", "relative_humidity"]
        ax = scatterviz(X[:,:2], y=y, ax=None, features=features)

        #test that is returns a matplotlib obj with axes
        self.assertIn('Axes', str(ax.properties()['axes']))

    @unittest.skipUnless(pandas is not None, "Pandas is not installed, could not run test.")
    def test_integrated_scatter_with_pandas(self):
        """
        Test scatter on the real, occupancy data set with a pandas dataframe
        """
        # Load the data from the fixture
        X = self.occupancy[[
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]]
        y = self.occupancy['occupancy'].astype(int)

        # Convert X to a pandas dataframe
        X = pandas.DataFrame(X)
        X.columns = ["temperature", "relative_humidity", "light", "C02", "humidity"]

        # Test the visualizer
        features = ["temperature", "relative_humidity"]
        visualizer = ScatterViz(features=features)
        visualizer.fit_transform_poof(X, y)
