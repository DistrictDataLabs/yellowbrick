# tests.test_features.test_projection
# Test the base ProjectionVisualizer drawing functionality
#
# Author:   Naresh Bachwani
# Created:  Wed Jul 17 09:53:07 2019 -0400
#
# Copyright (C) 2019 the scikit-yb developers.
# For license information, see LICENSE.txt
#
# ID: test_projection.py [21eb9d2] 43993586+naresh-bachwani@users.noreply.github.com $

"""
Test the base ProjectionVisualizer drawing functionality
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy.testing as npt
import matplotlib.pyplot as plt

from yellowbrick.features.projection import *
from yellowbrick.exceptions import YellowbrickValueError

from tests.base import VisualTestCase
from unittest import mock

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

##########################################################################
## MockVisualizer
##########################################################################


class MockVisualizer(ProjectionVisualizer):
    """
    The MockVisualizer implements the ProjectionVisualizer interface using
    PCA as an internal transformer. This visualizer is used to directly test
    how subclasses interact with the ProjectionVisualizer base class.
    """

    def __init__(
        self,
        ax=None,
        features=None,
        classes=None,
        colors=None,
        colormap=None,
        target_type="auto",
        projection=2,
        alpha=0.75,
        colorbar=True,
        **kwargs
    ):

        super(MockVisualizer, self).__init__(
            ax=ax,
            features=features,
            classes=classes,
            colors=colors,
            colormap=colormap,
            target_type=target_type,
            projection=projection,
            alpha=alpha,
            colorbar=colorbar,
            **kwargs
        )

        self.pca_transformer = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(self.projection, random_state=2019)),
            ]
        )

    def fit(self, X, y=None):
        super(MockVisualizer, self).fit(X, y)
        self.pca_transformer.fit(X)
        return self

    def transform(self, X, y=None):
        try:
            Xp = self.pca_transformer.transform(X)
        except AttributeError as e:
            raise AttributeError(str(e) + " try using fit_transform instead.")
        self.draw(Xp, y)
        return Xp


##########################################################################
## ProjectionVisualizer Tests
##########################################################################


@pytest.mark.usefixtures("discrete", "continuous")
class TestProjectionVisualizer(VisualTestCase):
    """
    Test the ProjectionVisualizer base class
    """

    def test_discrete_plot(self):
        """
        Test the visualizer with discrete target.
        """
        X, y = self.discrete
        classes = ["a", "b", "c", "d", "e"]
        visualizer = MockVisualizer(projection=2, colormap="plasma", classes=classes)
        X_prime = visualizer.fit_transform(X, y)
        npt.assert_array_equal(visualizer.classes_, classes)
        visualizer.finalize()
        self.assert_images_similar(visualizer)
        assert X_prime.shape == (self.discrete.X.shape[0], 2)

    def test_continuous_plot(self):
        """
        Test the visualizer with continuous target.
        """
        X, y = self.continuous
        visualizer = MockVisualizer(projection="2d")
        visualizer.fit_transform(X, y)
        visualizer.finalize()
        visualizer.cax.set_yticklabels([])
        self.assert_images_similar(visualizer)

    def test_continuous_when_target_discrete(self):
        """
        Ensure user can override discrete target_type by specifying continuous
        """
        _, ax = plt.subplots()
        X, y = self.discrete
        visualizer = MockVisualizer(
            ax=ax, projection="2D", target_type="continuous", colormap="cool"
        )
        visualizer.fit(X, y)
        visualizer.transform(X, y)
        visualizer.finalize()
        visualizer.cax.set_yticklabels([])
        self.assert_images_similar(visualizer)

    def test_single_plot(self):
        """
        Assert single color plot when y is not specified
        """
        X, y = self.discrete
        visualizer = MockVisualizer(projection=2, colormap="plasma")
        visualizer.fit_transform(X)
        visualizer.finalize()
        self.assert_images_similar(visualizer)

    def test_discrete_3d(self):
        """
        Test visualizer for 3 dimensional discrete plots
        """
        X, y = self.discrete

        classes = ["a", "b", "c", "d", "e"]
        colors = ["r", "b", "g", "m", "c"]
        visualizer = MockVisualizer(projection=3, colors=colors, classes=classes)
        visualizer.fit_transform(X, y)
        npt.assert_array_equal(visualizer.classes_, classes)
        visualizer.finalize()
        self.assert_images_similar(visualizer)

    def test_3d_continuous_plot(self):
        """
        Tests visualizer for 3 dimensional continuous plots
        """
        X, y = self.continuous
        visualizer = MockVisualizer(projection="3D")
        visualizer.fit_transform(X, y)
        visualizer.finalize()
        visualizer.cbar.set_ticks([])
        self.assert_images_similar(visualizer)

    def test_alpha_param(self):
        """
        Ensure that the alpha parameter modifies opacity
        """
        # Instantiate a prediction error plot, provide custom alpha
        X, y = self.discrete
        params = {"alpha": 0.3, "projection": 2}
        visualizer = MockVisualizer(**params)
        visualizer.ax = mock.MagicMock()
        visualizer.fit(X, y)
        visualizer.transform(X, y)

        assert visualizer.alpha == 0.3

        # Test that alpha was passed to internal matplotlib scatterplot
        _, scatter_kwargs = visualizer.ax.scatter.call_args
        assert "alpha" in scatter_kwargs
        assert scatter_kwargs["alpha"] == 0.3

    # Check Errors
    @pytest.mark.parametrize("projection", ["4D", 1, "100d", 0])
    def test_wrong_projection_dimensions(self, projection):
        """
        Validate projection hyperparameter
        """
        msg = "Projection dimensions must be either 2 or 3"
        with pytest.raises(YellowbrickValueError, match=msg):
            MockVisualizer(projection=projection)

    def test_target_not_label_encoded(self):
        """
        Assert label encoding mismatch with y raises exception
        """
        X, y = self.discrete
        # Multiply every element by 10 to make non-label encoded
        y = y * 10
        visualizer = MockVisualizer()
        msg = "Target needs to be label encoded."
        with pytest.raises(YellowbrickValueError, match=msg):
            visualizer.fit_transform(X, y)

    @pytest.mark.parametrize("dataset", ("discrete", "continuous"))
    def test_y_required_for_discrete_and_continuous(self, dataset):
        """
        Assert error is raised when y is not passed to transform
        """
        X, y = getattr(self, dataset)
        visualizer = MockVisualizer()
        visualizer.fit(X, y)

        msg = "y is required for {} target".format(dataset)
        with pytest.raises(YellowbrickValueError, match=msg):
            visualizer.transform(X)

    def test_colorbar_false(self):
        """
        Test that colorbar equals false works correctly
        """
        visualizer = MockVisualizer(colorbar=False, colormap="YlOrRd")
        visualizer.fit_transform(*self.continuous)
        visualizer.finalize()

        self.assert_images_similar(visualizer)
