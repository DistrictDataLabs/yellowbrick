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

import pytest

from yellowbrick.base import Visualizer
from yellowbrick.features.base import *
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
    
class DataVisualizerBaseTests(MultiFeatureVisualizer):
    
    def test_determine_target_color_type(self):
        """
        Check that the target type is determined by a value y
        """
        dataviz = DataVisualizer()
        # Check default is auto
        assert dataviz.target == AUTO

        # Assert single when y is None
        dataviz._determine_target_color_type(None)
        assert dataviz._target_color_type == SINGLE

        # Check when y is continuous
        y = np.random.rand(100)
        dataviz._determine_target_color_type(y)
        assert dataviz._target_color_type == CONTINUOUS

        # Check when y is discrete
        y = np.random.choice(['a', 'b', 'c', 'd'], 100)
        dataviz._determine_target_color_type(y)
        assert dataviz._target_color_type == DISCRETE

        # Check when default is set to continuous and discrete data passed in
        dataviz = dataviz(target=CONTINUOUS)
        y = np.random.choice(['a', 'b', 'c', 'd'], 100)
        dataviz._determine_target_color_type(y)
        assert dataviz._target_color_type == CONTINUOUS

        # Check when default is set to discrete and continuous data passed in
        dataviz = dataviz(target=DISCRETE)
        y = np.random.rand(100)
        dataviz._determine_target_color_type(y)
        assert dataviz._target_color_type == DISCRETE

        # None overrides specified target
        dataviz = dataviz(target=CONTINUOUS)
        dataviz._determine_target_color_type(None)
        assert dataviz._target_color_type == SINGLE

        # None overrides specified target
        dataviz = dataviz(target=DISCRETE)
        dataviz._determine_target_color_type(None)
        assert dataviz._target_color_type == SINGLE

        # Bad target raises exception
        # None overrides specified target
        dataviz = dataviz(target="foo")
        msg = "could not determine target color type"
        with pytest.raises(YellowbrickValueError, match=msg):
            dataviz._determine_target_color_type([])

