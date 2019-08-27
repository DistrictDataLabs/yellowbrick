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
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
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
    def test_poof_show_interface(self, mock_plt):
        """
        Test poof calls plt.show and other figure finalization correctly
        """

        class CustomVisualizer(Visualizer):
            pass

        _, ax = plt.subplots()
        viz = CustomVisualizer(ax=ax)
        viz.finalize = MagicMock()
        assert viz.poof() is ax

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
        assert viz.poof(outpath="test.png") is ax

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
            assert viz.poof() is not None


##########################################################################
## ModelVisualizer Cases
##########################################################################


class TestModelVisualizer(VisualTestCase):
    """
    Tests for the ModelVisualizer
    """

    def test_final_estimator(self):
        """
        Ensure the final estimator can be returned from a Pipeline or directly
        """
        svc = LinearSVC()
        assert ModelVisualizer(svc)._final_estimator() is svc

        short = Pipeline([("clf", svc)])
        assert ModelVisualizer(short)._final_estimator() is svc

        pipe = Pipeline([("pca", PCA()), ("clf", svc)])
        assert ModelVisualizer(pipe)._final_estimator() is svc

    @pytest.mark.parametrize(
        "model", [LinearSVC(), Pipeline([("pca", PCA()), ("clf", LinearSVC())])]
    )
    def test_get_learned_attr(self, model):
        """
        Ensure a learned attribute can be feteched from a model
        """
        X, y = load_occupancy(return_dataset=True).to_numpy()
        assert not hasattr(model, "classes_")

        viz = ModelVisualizer(model)
        assert viz.fit(X, y) is viz
        assert hasattr(model, "classes_")

        assert viz._get_learned_attr("classes_") is model.classes_

    @pytest.mark.parametrize(
        "model", [LinearSVC(), Pipeline([("pca", PCA()), ("clf", LinearSVC())])]
    )
    def test_get_learned_attr_not_fitted(self, model):
        """
        Assert NotFitted is raised on _get_learned_attr when not exists
        """
        assert not hasattr(model, "classes_")
        viz = ModelVisualizer(model)

        with pytest.raises(NotFitted, match="object has no attribute 'classes_'"):
            viz._get_learned_attr("classes_")


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
        # poof is required here (do not replace with finalize)!
        assert grid.poof() is not None

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
        # poof is required here (do not replace with finalize)!
        assert grid.poof() is not None

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
        # poof is required here (do not replace with finalize)!
        assert grid.poof() is not None

        self.assert_images_similar(grid)

    def test_cant_define_both_rows_cols(self):
        """
        Assert that both nrows and ncols cannot be specified
        """
        with pytest.raises(YellowbrickValueError):
            VisualizerGrid([], ncols=2, nrows=2)
