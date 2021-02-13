# tests.test_base.py
# Assertions for the base classes and abstract hierarchy.
#
# Author:   Rebecca Bilbro
# Author:   Benjamin Bengfort
# Author:   Neal Humphrey
# Created:  Sat Oct 08 18:34:30 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_base.py [83131ef] benjamin@bengfort.com $

"""
Assertions for the base classes and abstract hierarchy.
"""

##########################################################################
## Imports
##########################################################################

import pytest
import matplotlib.pyplot as plt

from yellowbrick.base import *
from yellowbrick.base import VisualizerGrid
from yellowbrick.datasets import load_occupancy
from yellowbrick.exceptions import YellowbrickWarning
from yellowbrick.exceptions import YellowbrickValueError

from unittest.mock import patch
from unittest.mock import MagicMock
from tests.rand import RandomVisualizer
from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase

from sklearn.svm import LinearSVC
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

    def test_visualizer_fig_property(self):
        """
        Test the fig property on the Visualizer
        """
        viz = Visualizer()
        assert viz._fig is None
        assert viz.fig is not None

        viz.fig = "foo"
        assert viz._fig == "foo"
        assert viz.fig == "foo"

    def test_size_property(self):
        """
        Test the size property on the base Visualizer
        """
        fig = plt.figure(figsize=(1, 2))
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
    def test_show_interface(self, mock_plt):
        """
        Test show calls plt.show and other figure finalization correctly
        """

        class CustomVisualizer(Visualizer):
            pass

        _, ax = plt.subplots()
        viz = CustomVisualizer(ax=ax)
        viz.finalize = MagicMock()
        assert viz.show() is ax

        viz.finalize.assert_called_once_with()
        mock_plt.show.assert_called_once_with()
        mock_plt.savefig.assert_not_called()

    @patch("yellowbrick.base.plt")
    def test_show_savefig_interface(self, mock_plt):
        """
        Test show calls plt.savefig and other figure finalization correctly
        """

        class CustomVisualizer(Visualizer):
            pass

        _, ax = plt.subplots()
        viz = CustomVisualizer(ax=ax)
        viz.finalize = MagicMock()
        assert viz.show(outpath="test.png") is ax

        viz.finalize.assert_called_once_with()
        mock_plt.show.assert_not_called()
        mock_plt.savefig.assert_called_once_with("test.png")

    @patch("yellowbrick.base.plt")
    def test_show_warns(self, mock_plt):
        """
        Test show issues a warning when no axes has been modified
        """

        class CustomVisualizer(Visualizer):
            pass

        with pytest.warns(YellowbrickWarning):
            viz = CustomVisualizer()
            assert viz.show() is not None

    def test_poof_deprecated(self):
        """
        Test that poof issues a deprecation warning
        """

        class CustomVisualizer(Visualizer):
            pass

        viz = CustomVisualizer()
        viz.show = MagicMock()

        with pytest.warns(DeprecationWarning, match="please use show"):
            viz.poof()

        viz.show.assert_called_once()


##########################################################################
## ModelVisualizer Cases
##########################################################################

class MockModelVisualizer(ModelVisualizer):

    def __init__(
        self,
        estimator,
        ax=None,
        fig=None,
        is_fitted="auto",
        param1='foo',
        param2='bar',
        **kwargs
    ):
        super().__init__(estimator, ax=ax, fig=fig, is_fitted=is_fitted, **kwargs)
        self.param1 = param1
        self.param2 = param2


class TestModelVisualizer(VisualTestCase):
    """
    Tests for the ModelVisualizer and wrapped estimator.
    """

    def test_get_params(self):
        """
        Test get_params from visualizer and wrapped estimator
        """
        model = LinearSVC()
        oz = MockModelVisualizer(model)
        params = oz.get_params()

        for key, val in model.get_params().items():
            assert key in params
            assert params[key] == val

        for key in ('ax', 'fig', 'is_fitted', 'param1', 'param2'):
            assert key in params

    def test_set_params(self):
        """
        Test set_params from visualizer to wrapped estimator
        """
        model = LinearSVC()
        oz = MockModelVisualizer(model)

        orig = oz.get_params()
        changes = {
            "param1": 'water',
            "is_fitted": False,
            "estimator__C": 8.8,
            "intercept_scaling": 4.2,
            "dual": False,
        }

        oz.set_params(**changes)
        params = oz.get_params()

        for key, val in params.items():
            if key in changes:
                assert val == changes[key]
            elif key == "C":
                assert val == changes["estimator__C"]
            else:
                assert val == orig[key]


##########################################################################
## ScoreVisualizer Cases
##########################################################################

class MockVisualizer(ScoreVisualizer):
    """
    Mock for a downstream score visualizer
    """

    def fit(self, X, y):
        super(MockVisualizer, self).fit(X, y)


class TestScoreVisualizer(VisualTestCase):
    """
    Tests for the ScoreVisualizer
    """

    def test_with_fitted(self):
        """
        Test that visualizer properly handles an already-fitted model
        """
        X, y = load_occupancy(return_dataset=True).to_numpy()

        model = LinearSVC().fit(X, y)
        classes = ["unoccupied", "occupied"]

        with patch.object(model, "fit") as mockfit:
            oz = MockVisualizer(model, classes=classes)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with patch.object(model, "fit") as mockfit:
            oz = MockVisualizer(model, classes=classes, is_fitted=True)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with patch.object(model, "fit") as mockfit:
            oz = MockVisualizer(model, classes=classes, is_fitted=False)
            oz.fit(X, y)
            mockfit.assert_called_once_with(X, y)


##########################################################################
## Visual Grid Cases
##########################################################################


@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")
class TestVisualizerGrid(VisualTestCase):
    """
    Tests for the VisualizerGrid layout class
    """

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
    )
    def test_draw_visualizer_grid(self):
        """
        Draw a 4 visualizers grid with default options
        """
        visualizers = [RandomVisualizer(random_state=(1 + x) ** 2) for x in range(4)]

        X, y = make_classification(random_state=78)
        grid = VisualizerGrid(visualizers)

        grid.fit(X, y)
        # show is required here (do not replace with finalize)!
        assert grid.show() is not None

        self.assert_images_similar(grid)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
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
        # show is required here (do not replace with finalize)!
        assert grid.show() is not None

        self.assert_images_similar(grid)

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="font rendering different in OS and/or Python; see #892",
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
        # show is required here (do not replace with finalize)!
        assert grid.show() is not None

        self.assert_images_similar(grid)

    def test_cant_define_both_rows_cols(self):
        """
        Assert that both nrows and ncols cannot be specified
        """
        with pytest.raises(YellowbrickValueError):
            VisualizerGrid([], ncols=2, nrows=2)
