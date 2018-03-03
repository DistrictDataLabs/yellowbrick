# tests.test_classifier.test_threshold
# Ensure that the threshold visualizations work.
#
# Author:   Nathan Danielsen <ndanielsen@gmail.com.com>
# Created:  Wed April 26 20:17:29 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_threshold.py [] nathan.danielsen@gmail.com $
"""
Ensure that the Threshold visualizations work.
"""

##########################################################################
## Imports
##########################################################################
import unittest
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin
from yellowbrick.classifier import *

from sklearn.naive_bayes import BernoulliNB

##########################################################################
## Data
##########################################################################

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

##########################################################################
## Threshold visualizer test case
##########################################################################


class ThresholdVisualizerTest(VisualTestCase, DatasetMixin):
    def setUp(self):
        self.occupancy = self.load_data('occupancy')
        super(ThresholdVisualizerTest, self).setUp()

    def tearDown(self):
        self.occupancy = None
        super(ThresholdVisualizerTest, self).tearDown()

    def test_threshold_vi__init__(self):
        """
        Test that init params are in order
        """
        model = BernoulliNB(3)
        viz = ThresholdVisualizer(model)
        self.assertIs(viz.estimator, model)
        self.assertIsNone(viz.color)
        self.assertIsNone(viz.title)
        self.assertIsNone(viz.plot_data)
        self.assertEquals(viz.n_trials, 50)
        self.assertEquals(viz.test_size_percent, 0.1)
        self.assertEquals(viz.quantiles, (0.1, 0.5, 0.9))


    def test_threshold_viz(self):
        """
        Integration test of threshold visualizers
        """
        model = BernoulliNB(3)
        visualizer = ThresholdVisualizer(model, random_state=0)

        # assert that fit method returns ax
        self.assertIs(visualizer.ax, visualizer.fit(X, y=y))

        visualizer.poof()

        self.assert_images_similar(visualizer)

    @unittest.skipUnless(pd is not None, "Pandas is not installed, could not run test.")
    def test_threshold_viz_read_data(self):
        """
        Test ThresholdVisualizer on the real, occupancy data set with pandas
        """
        # Load the data from the fixture
        X = self.occupancy[[
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]]
        y = self.occupancy['occupancy'].astype(int)

        # Convert X to a pandas dataframe
        X = pd.DataFrame(X)
        X.columns = [
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]

        model = BernoulliNB(3)
        visualizer = ThresholdVisualizer(model, random_state=0)
        # Fit and transform the visualizer (calls draw)
        visualizer.fit(X, y)
        visualizer.draw()
        visualizer.poof()
        self.assert_images_similar(visualizer)

    def test_threshold_viz_quick_method_read_data(self):
        """
        Test for thresholdviz quick method with visual unit test
        """
        model = BernoulliNB(3)

        visualizer = type('Visualizer', (object, ),
                          {'ax': thresholdviz(model, X, y)})
        self.assert_images_similar(visualizer)
