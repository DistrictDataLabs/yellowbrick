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
import unittest
from unittest.mock import patch

from yellowbrick.base import Visualizer
from yellowbrick.features.base import *
from tests.base import VisualTestCase
from ..fixtures import TestDataset

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_classification, make_regression

##########################################################################
## Fixtures
##########################################################################

@pytest.fixture(scope='class')
def discrete(request):
    """
    Creare a random classification fixture.
    """
    X, y = make_classification(
        n_samples=400, n_features=12, n_informative=10, n_redundant=0, 
        n_classes=5, random_state=2019)

    # Set a class attribute for digits
    request.cls.discrete = TestDataset(X, y)

@pytest.fixture(scope='class')
def continuous(request):
    """
    Creates a random regressor fixture. 
    """
    X, y = make_regression(
        n_samples=500, n_features=22, n_informative=8, random_state=2019
    )

    # Set a class attribute for digits
    request.cls.continuous = TestDataset(X, y)


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


@pytest.mark.usefixtures("discrete", "continuous")
class DataVisualizerBaseTests(unittest.TestCase):
    
    @patch.object(DataVisualizer, 'draw')
    def test_single(self, mock_draw):

        dataviz = DataVisualizer()
        # Check default is auto
        assert dataviz.target == target_type.AUTO

        # Assert single when y is None
        dataviz._determine_target_color_type(None)
        assert dataviz._target_color_type == target_type.SINGLE
        
        # None overrides specified target
        dataviz = DataVisualizer(target="continuous")
        X, y = self.continuous
        dataviz.fit(X)
        mock_draw.assert_called_once()
        assert dataviz._colors == 'b'
        assert dataviz._target_color_type == target_type.SINGLE
       
        # None overrides specified target
        dataviz = DataVisualizer(target="discrete")
        dataviz._determine_target_color_type(None)
        assert dataviz._target_color_type == target_type.SINGLE

    @patch.object(DataVisualizer, 'draw')
    def test_continuous(self, mock_draw):
        # Check when y is continuous
        X, y = self.continuous
        dataviz = DataVisualizer()
        dataviz.fit(X, y)
        mock_draw.assert_called_once()
        assert hasattr(dataviz, "range_")
        assert dataviz._target_color_type == target_type.CONTINUOUS
        
        # Check when default is set to continuous and discrete data passed in
        dataviz = DataVisualizer(target="continuous")
        X, y = self.discrete
        dataviz._determine_target_color_type(y)
        assert dataviz._target_color_type == target_type.CONTINUOUS
        
    def test_discrete(self):
        # Check when y is discrete
        _, y = self.discrete
        dataviz = DataVisualizer()
        dataviz._determine_target_color_type(y)
        assert dataviz._target_color_type == target_type.DISCRETE
        
        # Check when default is set to discrete and continuous data passed in
        _, y = self.continuous
        dataviz = DataVisualizer(target="discrete")
        dataviz._determine_target_color_type(y)
        assert dataviz._target_color_type == target_type.DISCRETE
        
    def test_bad_target(self):
        # Bad target raises exception
        # None overrides specified target
        msg = "unknown target color type 'foo'"
        with pytest.raises(YellowbrickValueError, match=msg):
            DataVisualizer(target="foo")
    
    @patch.object(DataVisualizer, 'draw')        
    def test_classes(self, mock_draw):
        # Checks that classes are assigned correctly
        X, y = self.discrete
        classes = ['a', 'b', 'c', 'd', 'e']
        dataviz = DataVisualizer(classes=classes, target='discrete')
        dataviz.fit(X, y)
        assert dataviz.classes_ == classes
        assert list(dataviz._colors.keys()) == classes
        
        