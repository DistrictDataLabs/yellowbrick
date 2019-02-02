# tests.test_text.test_base
# Tests for the text visualization base classes

"""
Tests for the text visualization base classes
"""

##########################################################################
## Imports
##########################################################################

import unittest

from yellowbrick.base import *
from yellowbrick.gridsearch.base import *
from sklearn.base import BaseEstimator, TransformerMixin


##########################################################################
## TextVisualizer Base Tests
##########################################################################

class GridSearchVisualizerBaseTests(unittest.TestCase):

    def test_subclass(self):
        """
        Assert the Grid Search visualizer is subclassed correctly 
        """
        visualizer = GridSearchVisualizer()
        self.assertIsInstance(visualizer, TransformerMixin)
        self.assertIsInstance(visualizer, BaseEstimator)
        self.assertIsInstance(visualizer, ModelVisualizer)
