# tests.test_base.py
# Assertions for the base classes and abstract hierarchy.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Author:   Neal Humphrey
# Created:  Sat Oct 08 18:34:30 2016 -0400
#
# ID: test_base.py [83131ef] benjamin@bengfort.com $

"""
Assertions for the base classes and abstract hierarchy.
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest
import matplotlib.pyplot as plt

from yellowbrick.base import *
from yellowbrick.base import VisualizerGrid
from yellowbrick.exceptions import YellowbrickWarning
from yellowbrick.exceptions import YellowbrickValueError

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.base import VisualTestCase
from tests.rand import RandomVisualizer


from sklearn.datasets import make_classification


##########################################################################
## Base Cases
##########################################################################

class TestBaseClasses(VisualTestCase):
    """
    Tests for the high-level API of Yellowbrick and base classes
    """

    def test_visualizer_ax_property(self):
        """
        Test the ax property on the Visualizer
        """
        viz = Visualizer()
        assert viz._ax is None
        assert viz.ax is not None

        viz.ax = "foo"
        assert viz._ax == "foo"
        assert viz.ax == "foo"

    def test_size_property(self):
        """
        Test the size property on the base Visualizer
        """
        fig = plt.figure(figsize =(1,2))
        viz = Visualizer()

        assert viz._size is None
        assert viz.size is not None

        fsize = fig.get_size_inches() * fig.get_dpi()
        assert all(viz.size) == all(fsize)

        viz.size = (1080, 720)
        assert viz._size == (1080, 720)
        assert viz.size == (1080, 720)

        fsize = fig.get_size_inches() * fig.get_dpi()
        assert all(viz.size) == all(fsize)

    def test_visualizer_fit_returns_self(self):
        """
        Assert that all visualizers return self
        """
        viz = Visualizer()
        assert viz.fit([]) is viz

    def test_draw_interface(self):
        """
        Assert that draw cannot be called at the base level
        """
        with pytest.raises(NotImplementedError):
            viz = Visualizer()
            viz.draw()

    def test_finalize_interface(self):
        """
        Assert finalize returns the finalized axes
        """
        viz = Visualizer()
        assert viz.finalize() is viz.ax

    @patch("yellowbrick.base.plt")
    def test_poof_show_interface(self, mock_plt):
        """
        Test poof calls plt.show and other figure finalization correctly
        """
        class CustomVisualizer(Visualizer):
            pass

        _, ax = plt.subplots()
        viz = CustomVisualizer(ax=ax)
        viz.finalize = MagicMock()
        viz.poof()

        viz.finalize.assert_called_once_with()
        mock_plt.show.assert_called_once_with()
        mock_plt.savefig.assert_not_called()

    @patch("yellowbrick.base.plt")
    def test_poof_savefig_interface(self, mock_plt):
        """
        Test poof calls plt.savefig and other figure finalization correctly
        """
        class CustomVisualizer(Visualizer):
            pass

        _, ax = plt.subplots()
        viz = CustomVisualizer(ax=ax)
        viz.finalize = MagicMock()
        viz.poof(outpath="test.png")

        viz.finalize.assert_called_once_with()
        mock_plt.show.assert_not_called()
        mock_plt.savefig.assert_called_once_with("test.png")

    @patch("yellowbrick.base.plt")
    def test_poof_warns(self, mock_plt):
        """
        Test poof issues a warning when no axes has been modified
        """
        class CustomVisualizer(Visualizer):
            pass

        with pytest.warns(YellowbrickWarning):
            viz = CustomVisualizer()
            viz.poof()


##########################################################################
## Visual Grid Cases
##########################################################################

@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")
class TestVisualizerGrid(VisualTestCase):
    """
    Tests for the VisualizerGrid layout class
    """

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_draw_visualizer_grid(self):
        """
        Draw a 4 visualizers grid with default options
        """
        visualizers = [
            RandomVisualizer(random_state=(1+x)**2)
            for x in range(4)
        ]

        X, y = make_classification(random_state=78)
        grid = VisualizerGrid(visualizers)

        grid.fit(X, y)
        grid.poof() # poof is required here (do not replace with finalize)!

        self.assert_images_similar(grid)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_draw_with_rows(self):
        """
        Draw 2 visualizers in their own row
        """
        visualizers = [
            RandomVisualizer(random_state=63),
            RandomVisualizer(random_state=36),
        ]

        X, y = make_classification(random_state=87)
        grid = VisualizerGrid(visualizers, nrows=2)

        grid.fit(X, y)
        grid.poof() # poof is required here (do not replace with finalize)!

        self.assert_images_similar(grid)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_draw_with_cols(self):
        """
        Draw 2 visualizers in their own column
        """
        visualizers = [
            RandomVisualizer(random_state=633),
            RandomVisualizer(random_state=336),
        ]

        X, y = make_classification(random_state=187)
        grid = VisualizerGrid(visualizers, ncols=2)

        grid.fit(X, y)
        grid.poof() # poof is required here (do not replace with finalize)!

        self.assert_images_similar(grid)

    def test_cant_define_both_rows_cols(self):
        """
        Assert that both nrows and ncols cannot be specified
        """
        with pytest.raises(YellowbrickValueError):
            VisualizerGrid([], ncols=2, nrows=2)
