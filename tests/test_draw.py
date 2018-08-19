# tests.test_draw
# Tests for the high-level drawing utility functions
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Sun Aug 19 11:21:04 2018 -0400
#
# ID: test_draw.py [] benjamin@bengfort.com $

"""
Tests for the high-level drawing utility functions
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np
import matplotlib.pyplot as plt

from yellowbrick.draw import *
from .base import VisualTestCase


##########################################################################
## Simple tests for high-level drawing utilities
##########################################################################

def test_manual_legend_uneven_colors():
    """
    Raise exception when colors and labels are mismatched in manual_legend
    """
    with pytest.raises(YellowbrickValueError, match="same number of colors as labels"):
        manual_legend(None, ('a', 'b', 'c'), ('r', 'g'))


##########################################################################
## Visual test cases for high-level drawing utilities
##########################################################################

class TestDraw(VisualTestCase):
    """
    Visual tests for the high-level drawing utilities
    """

    def test_manual_legend(self):
        """
        Check that the manual legend is drawn without axes artists
        """
        # Draw a random scatter plot
        random = np.random.RandomState(42)

        Ax, Ay = random.normal(50, 2, 100), random.normal(50, 3, 100)
        Bx, By = random.normal(42, 3, 100), random.normal(44, 1, 100)
        Cx, Cy = random.normal(20, 10, 100), random.normal(30, 1, 100)


        _, ax = plt.subplots()
        ax.scatter(Ax, Ay, c='r', alpha=0.35, label='a')
        ax.scatter(Bx, By, c='g', alpha=0.35, label='b')
        ax.scatter(Cx, Cy, c='b', alpha=0.35, label='c')

        # Add the manual legend
        manual_legend(
            ax, ('a', 'b', 'c'), ('r', 'g', 'b'), frameon=True, loc='upper left'
        )

        # Assert image similarity
        self.assert_images_similar(ax=ax)
