# tests.test_text.test_base
# Tests for the text visualization base classes
#
# Author:   Benjamin Bengfort
# Created:  Mon Feb 20 06:34:50 2017 -0500
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_base.py [6aa9198] benjamin@bengfort.com $

"""
Tests for the text visualization base classes
"""

##########################################################################
## Imports
##########################################################################

from yellowbrick.base import *
from yellowbrick.text.base import *
from sklearn.base import BaseEstimator, TransformerMixin


##########################################################################
## TextVisualizer Base Tests
##########################################################################


class TestTextVisualizerBase(object):
    def test_subclass(self):
        """
        Assert the text visualizer is subclassed correctly
        """
        visualizer = TextVisualizer()
        assert isinstance(visualizer, TransformerMixin)
        assert isinstance(visualizer, BaseEstimator)
        assert isinstance(visualizer, Visualizer)
