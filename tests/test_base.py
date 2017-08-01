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
