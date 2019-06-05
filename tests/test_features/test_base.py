# tests.test_features.test_base
# Tests for the feature selection and analysis base classes
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Oct 07 13:43:55 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_base.py [2e898a6] benjamin@bengfort.com $

"""
Tests for the feature selection and analysis base classes
"""

##########################################################################
## Imports
##########################################################################

from yellowbrick.base import Visualizer
from yellowbrick.features.base import FeatureVisualizer
from tests.base import VisualTestCase

from sklearn.base import BaseEstimator, TransformerMixin


##########################################################################
## FeatureVisualizer Base Tests
##########################################################################

class FeatureVisualizerBaseTests(VisualTestCase):

    def test_subclass(self):
        """
        Assert the feature visualizer is in its rightful place
        """
        visualizer = FeatureVisualizer()
        self.assertIsInstance(visualizer, TransformerMixin)
        self.assertIsInstance(visualizer, BaseEstimator)
        self.assertIsInstance(visualizer, Visualizer)

    # def test_interface(self):
    #     """
    #     Test the feature visualizer interface
    #     """
    #
    #     visualizer = FeatureVisualizer()
    #     with self.assertRaises(NotImplementedError):
    #         visualizer.poof()
