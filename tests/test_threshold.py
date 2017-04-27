# tests.test_threshold
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

from unittest import mock
import unittest
import numpy as np

from tests.base import VisualTestCase
from yellowbrick.threshold import *

from sklearn.naive_bayes import BernoulliNB


##########################################################################
## Data
##########################################################################

X = np.array(
        [[ 2.318, 2.727, 4.260, 7.212, 4.792],
         [ 2.315, 2.726, 4.295, 7.140, 4.783,],
         [ 2.315, 2.724, 4.260, 7.135, 4.779,],
         [ 2.110, 3.609, 4.330, 7.985, 5.595,],
         [ 2.110, 3.626, 4.330, 8.203, 5.621,],
         [ 2.110, 3.620, 4.470, 8.210, 5.612,],
         [ 2.318, 2.727, 4.260, 7.212, 4.792,],
         [ 2.315, 2.726, 4.295, 7.140, 4.783,],
         [ 2.315, 2.724, 4.260, 7.135, 4.779,],
         [ 2.110, 3.609, 4.330, 7.985, 5.595,],
         [ 2.110, 3.626, 4.330, 8.203, 5.621,],
         [ 2.110, 3.620, 4.470, 8.210, 5.612,]]
    )

y = np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])


##########################################################################
## Threshold visualizer test case
##########################################################################

class ThresholdVisualizerTest(VisualTestCase):

    def test_threshold_viz(self):
        """
        Assert no errors occur during KnnDecisionBoundariesVisualizer integration
        """
        model = BernoulliNB(3)
        viz = ThresholdVisualizer(model)
        viz.fit_draw_poof(X, y=y)
