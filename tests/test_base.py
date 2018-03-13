# tests.test_base.py
# Assertions for the base classes and abstract hierarchy.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Sat Oct 08 18:34:30 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_base.py [83131ef] benjamin@bengfort.com $

"""
Assertions for the base classes and abstract hierarchy.
"""

##########################################################################
## Imports
##########################################################################

import unittest

from yellowbrick.base import *

try:
    from unittest import mock
except ImportError:
    import mock

# Transitioning to pytest
import pytest

##########################################################################
## Imports
##########################################################################

class BaseTests(unittest.TestCase):
    """
    Test the high level API for yellowbrick
    """

    def test_ax_property(self):
        """
        Test the ax property on the base Visualizer
        """
        visualizer = Visualizer()
        self.assertIsNone(visualizer._ax)
        self.assertIsNotNone(visualizer.ax)

        visualizer.ax = "foo"
        self.assertEqual(visualizer._ax, "foo")
        self.assertEqual(visualizer.ax, "foo")

    def test_visualizer_fit_returns_self(self):
        """
        Assert that all visualizers return self
        """
        visualizer = Visualizer()
        self.assertIs(visualizer.fit([]), visualizer)

    def test_draw_interface(self):
        """
        Assert that draw cannot be called at the base level
        """
        with self.assertRaises(NotImplementedError):
            visualizer = Visualizer()
            visualizer.draw()

    def test_finalize_interface(self):
        """
        Assert finalize returns the finalized axes
        """
        visualizer = Visualizer()
        self.assertIs(visualizer.finalize(), visualizer.ax)

    def test_size_property(self):
        """
        Test the size property on the base Visualizer
        """
        fig = plt.figure(figsize =(1,2))
        visualizer = Visualizer()
        self.assertIsNone(visualizer._size)
        self.assertIsNotNone(visualizer.size)
        figure_size = fig.get_size_inches() * fig.get_dpi()
        self.assertEqual(all(visualizer.size), all(figure_size))
        visualizer.size = (1080, 720)
        figure_size = fig.get_size_inches() * fig.get_dpi()
        self.assertEqual(all(visualizer.size), all(figure_size))
        self.assertEqual(visualizer._size, (1080, 720))
        self.assertEqual(visualizer.size, (1080, 720))




##########################################
# MultipleVisualizer
##########################################
from yellowbrick.features.radviz import RadViz
from yellowbrick.base import MultipleVisualizer
from yellowbrick.exceptions import YellowbrickValueError

import numpy as np

class TestExample():
    def setup_method(self,method):
        self.X = np.array([[10,20,30],[5,10,15],[20,30,40],[2,3,4],[10,20,5]])
        self.y = np.array([1,0,1,1,0])
        self.classes = ["A","B"]
        self.features = ["first","second","third"]
        self.visualizers = [RadViz(classes=self.classes, features=self.features),
                       RadViz(classes=self.classes, features=self.features)
                      ]
    def test_draw_multiplevisualizer(self):
        #A simple multiple visualizer that puts two RadViz on two subplots
        mv = MultipleVisualizer(self.visualizers)
        mv.fit(self.X,self.y)
        mv.poof()

    def test_draw_with_rows(self):
        #A simple multiple visualizer that puts two RadViz on two subplots
        mv = MultipleVisualizer(self.visualizers, nrows=2)
        mv.fit(self.X,self.y)
        mv.poof()

    def test_draw_with_cols(self):
        #A simple multiple visualizer that puts two RadViz on two subplots
        mv = MultipleVisualizer(self.visualizers, ncols=2)
        mv.fit(self.X,self.y)
        mv.poof()

    def test_cant_define_both_rows_cols(self):
        with pytest.raises(YellowbrickValueError) as e:
            mv = MultipleVisualizer(self.visualizers, ncols=2, nrows=2)