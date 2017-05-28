# tests.test_text.test_base
# Tests for the text visualization base classes
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Mon Feb 20 06:34:50 2017 -0500
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_base.py [6aa9198] benjamin@bengfort.com $

"""
Tests for the text visualization base classes
"""

##########################################################################
## Imports
##########################################################################

import unittest

from yellowbrick.base import *
from yellowbrick.text.base import *
from sklearn.base import BaseEstimator, TransformerMixin


##########################################################################
## TextVisualizer Base Tests
##########################################################################

class TextVisualizerBaseTests(unittest.TestCase):

    def test_subclass(self):
        """
        Assert the text visualizer is subclassed correctly 
        """
        visualizer = TextVisualizer()
        self.assertIsInstance(visualizer, TransformerMixin)
        self.assertIsInstance(visualizer, BaseEstimator)
        self.assertIsInstance(visualizer, Visualizer)

    # def test_interface(self):
    #     """
    #     Test the feature visualizer interface
    #     """
    #
    #     visualizer = TextVisualizer()
    #     with self.assertRaises(NotImplementedError):
    #         visualizer.poof()
