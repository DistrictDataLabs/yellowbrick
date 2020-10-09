# tests.test_draw
# Tests for the high-level drawing utility functions
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Sun Aug 19 11:21:04 2018 -0400
#
# ID: test_draw.py [dd915ad] benjamin@bengfort.com $

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
        manual_legend(None, ("a", "b", "c"), ("r", "g"))


@pytest.fixture(scope="class")
def data(request):

    data = np.array(
        [
            [4, 8, 7, 6, 5, 2, 1],
            [6, 7, 9, 6, 9, 3, 6],
            [5, 1, 6, 8, 4, 7, 8],
            [6, 8, 1, 5, 6, 7, 4],
        ]
    )

    request.cls.data = data


##########################################################################
## Visual test cases for high-level drawing utilities
##########################################################################


@pytest.mark.usefixtures("data")
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
        ax.scatter(Ax, Ay, c="r", alpha=0.35, label="a")
        ax.scatter(Bx, By, c="g", alpha=0.35, label="b")
        ax.scatter(Cx, Cy, c="b", alpha=0.35, label="c")

        # Add the manual legend
        manual_legend(
            ax, ("a", "b", "c"), ("r", "g", "b"), frameon=True, loc="upper left"
        )

        # Assert image similarity
        self.assert_images_similar(ax=ax, tol=0.5)

    def test_vertical_bar_stack(self):
        """
        Test bar_stack for vertical orientation
        """
        _, ax = plt.subplots()

        # Plots stacked bar charts
        bar_stack(self.data, ax=ax, orientation="v")

        # Assert image similarity
        self.assert_images_similar(ax=ax, tol=0.1)

    def test_horizontal_bar_stack(self):
        """
        Test bar_stack for horizontal orientation
        """
        _, ax = plt.subplots()
        # Plots stacked bar charts
        bar_stack(self.data, ax=ax, orientation="h")

        # Assert image similarity
        self.assert_images_similar(ax=ax, tol=0.1)

    def test_single_row_bar_stack(self):
        """
        Test bar_stack for single row
        """
        data = np.array([[4, 8, 7, 6, 5, 2, 1]])

        _, ax = plt.subplots()

        # Plots stacked bar charts
        bar_stack(data, ax=ax)

        # Assert image similarity
        self.assert_images_similar(ax=ax, tol=0.1)

    def test_labels_vertical(self):
        """
        Test labels and ticks for vertical barcharts
        """
        labels = ["books", "cinema", "cooking", "gaming"]
        ticks = ["noun", "verb", "adverb", "pronoun", "preposition", "digit", "other"]
        _, ax = plt.subplots()

        # Plots stacked bar charts
        bar_stack(self.data, labels=labels, ticks=ticks, colors=["r", "b", "g", "y"])

        # Extract tick labels from the plot
        ticks_ax = [tick.get_text() for tick in ax.xaxis.get_ticklabels()]
        # Assert that ticks are set properly
        assert ticks_ax == ticks

        # Assert image similarity
        self.assert_images_similar(ax=ax, tol=0.05)

    def test_labels_horizontal(self):
        """
        Test labels and ticks with horizontal barcharts
        """
        labels = ["books", "cinema", "cooking", "gaming"]
        ticks = ["noun", "verb", "adverb", "pronoun", "preposition", "digit", "other"]
        _, ax = plt.subplots()

        # Plots stacked bar charts
        bar_stack(
            self.data, labels=labels, ticks=ticks, orientation="h", colormap="cool"
        )

        # Extract tick labels from the plot
        ticks_ax = [tick.get_text() for tick in ax.yaxis.get_ticklabels()]
        # Assert that ticks are set properly
        assert ticks_ax == ticks

        # Assert image similarity
        self.assert_images_similar(ax=ax, tol=0.05)
