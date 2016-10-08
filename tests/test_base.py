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
# ID: tests.test_base.py.py [] benjamin@bengfort.com $

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

    def test_visualizer_returns_self(self):
        """
        Assert that all visualizers return self
        """
        visualizer = Visualizer()
        self.assertIs(visualizer.fit([]), visualizer)

    def test_base_poof(self):
        """
        Assert that the base visualizer implements poof interface
        """
        with self.assertRaises(NotImplementedError):
            visualizer = Visualizer()
            visualizer.poof()

    def test_fit_draw(self):
        """
        Assert fit_draw calls fit and draw
        """

        visualizer = Visualizer()
        visualizer.fit = mock.Mock()
        visualizer.draw = mock.Mock()

        visualizer.fit_draw([])

        visualizer.fit.assert_called_once_with([], None)
        visualizer.draw.assert_called_once_with()
