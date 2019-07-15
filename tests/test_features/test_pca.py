# -*- coding: utf-8 -*-
# tests.test_features.test_pca
# Tests for the PCA based feature visualizer.
#
# Author:   Carlo Morales <@cjmorale>
# Author:   Raúl Peralta Lozada <@RaulPL>
# Author:   Benjamin Bengfort <@bbengfort>
# Created:  Tue May 23 18:34:27 2017 -0400
#
# ID: test_pca.py [] cmorales@pacificmetrics.com $

"""
Tests for the PCA based feature visualizer.
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest
import numpy as np

from unittest import mock
from ..fixtures import Dataset
from tests.base import VisualTestCase

from yellowbrick.features.pca import *
from yellowbrick.exceptions import YellowbrickError

from sklearn.datasets import make_classification

try:
    # Only available in Matplotlib >= 2.0.2
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    make_axes_locatable = None

##########################################################################
## Fixtures
##########################################################################

@pytest.fixture(scope='class')
def binary(request):
    """
    Creates a fixture of train and test splits for the sklearn digits dataset
    For ease of use returns a Dataset named tuple composed of two Split tuples.
    """
    X, y = make_classification(
        n_samples=400, n_features=12, n_informative=8, n_redundant=0,
        n_classes=2, n_clusters_per_class=1, class_sep=1.8, random_state=854,
        scale=[14.2, 2.1, 0.32, 0.001, 32.3, 44.1, 102.3, 2.3, 2.4, 38.2, 0.05, 1.0],
    )

    # Set a class attribute for digits
    request.cls.dataset = Dataset(X, y)


##########################################################################
##PCA Tests
##########################################################################

@pytest.mark.usefixtures("binary")
class TestPCADecomposition(VisualTestCase):
    """
    Test the PCADecomposition visualizer
    """

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows (RMSE=4)"
    )
    def test_pca_decomposition_quick_method(self):
        """
        Test the quick method PCADecomposition visualizer 2 dimensions scaled.
        """
        ax = pca_decomposition(
            X=self.dataset.X, proj_dim=2, scale=True, random_state=28
        )
        self.assert_images_similar(ax=ax, tol=5)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows (RMSE=?)"
    )
    def test_scale_true_2d(self):
        """
        Test the PCADecomposition visualizer 2 dimensions scaled.
        """
        params = {'scale': True, 'proj_dim': 2, 'random_state': 9932}
        visualizer = PCADecomposition(**params).fit(self.dataset.X)
        pca_array = visualizer.transform(self.dataset.X)

        # Image comparison tests
        self.assert_images_similar(visualizer)

        # Assert PCA transformation occurred successfully
        assert pca_array.shape == (self.dataset.X.shape[0], 2)

    def test_scale_false_2d(self):
        """
        Test the PCADecomposition visualizer 2 dimensions non-scaled.
        """
        params = {'scale': False, 'proj_dim': 2, 'random_state': 1229}
        visualizer = PCADecomposition(**params).fit(self.dataset.X)
        pca_array = visualizer.transform(self.dataset.X)

        # Image comparison tests
        self.assert_images_similar(visualizer, tol=0.03)

        # Assert PCA transformation occurred successfully
        assert pca_array.shape == (self.dataset.X.shape[0], 2)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows (RMSE=3)"
    )
    def test_biplot_2d(self):
        """
        Test the PCADecomposition 2D biplot (proj_features).
        """
        params = {
            'features': 'ABCDEFGHIKLM', 'random_state': 67,
            'proj_features': True, 'proj_dim': 2,
        }
        visualizer = PCADecomposition(**params).fit(self.dataset.X)
        pca_array = visualizer.transform(self.dataset.X)

        # Image comparison tests
        self.assert_images_similar(visualizer, tol=5)

        # Assert PCA transformation occurred successfully
        assert pca_array.shape == (self.dataset.X.shape[0], 2)

    def test_scale_true_3d(self):
        """
        Test the PCADecomposition visualizer 3 dimensions scaled.
        """
        params = {'scale': True, 'proj_dim': 3, 'random_state': 7382}
        visualizer = PCADecomposition(**params).fit(self.dataset.X)
        pca_array = visualizer.transform(self.dataset.X)

        # Image comparison tests
        self.assert_images_similar(visualizer)

        # Assert PCA transformation occurred successfully
        assert pca_array.shape == (self.dataset.X.shape[0], 3)

    def test_scale_false_3d(self):
        """
        Test the PCADecomposition visualizer 3 dimensions non-scaled.
        """
        params = {'scale': False, 'proj_dim': 3, 'random_state': 98}
        visualizer = PCADecomposition(**params).fit(self.dataset.X)
        pca_array = visualizer.transform(self.dataset.X)

        # Image comparison tests
        self.assert_images_similar(visualizer)

        # Assert PCA transformation occurred successfully
        assert pca_array.shape == (self.dataset.X.shape[0], 3)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows (RMSE=3)"
    )
    def test_biplot_3d(self):
        """
        Test the PCADecomposition 3D biplot (proj_features).
        """
        params = {
            'features': 'ABCDEFGHIKLM', 'random_state': 800,
            'proj_features': True, 'proj_dim': 3,
        }
        visualizer = PCADecomposition(**params).fit(self.dataset.X)
        pca_array = visualizer.transform(self.dataset.X)

        # Image comparison tests
        self.assert_images_similar(visualizer, tol=5)

        # Assert PCA transformation occurred successfully
        assert pca_array.shape == (self.dataset.X.shape[0], 3)

    def test_scale_true_4d_execption(self):
        """
        Test the PCADecomposition visualizer 4 dimensions scaled (catch YellowbrickError).
        """
        params = {'scale': True, 'proj_dim': 4}
        with pytest.raises(YellowbrickError, match="proj_dim object is not 2 or 3"):
            PCADecomposition(**params)

    def test_scale_true_3d_execption(self):
        """
        Test the PCADecomposition visualizer 3 dims scaled on 2 dim data set (catch ValueError).
        """
        X = np.random.normal(loc=2, size=(100, 2))
        params = {'scale': True, 'proj_dim': 3}

        e = r'n_components=3 must be between 0 and min\(n_samples, n_features\)=2'
        with pytest.raises(ValueError, match=e):
            pca = PCADecomposition(**params)
            pca.fit(X)

    @mock.patch('yellowbrick.features.pca.plt.sca', autospec=True)
    def test_alpha_param(self, mock_sca):
        """
        Test that the user can supply an alpha param on instantiation
        """
        # Instantiate a prediction error plot, provide custom alpha
        params = {'alpha': 0.3, 'proj_dim': 2, 'random_state': 9932}
        visualizer = PCADecomposition(**params).fit(self.dataset.X)
        pca_array = visualizer.transform(self.dataset.X)
        assert visualizer.alpha == 0.3

        visualizer.ax = mock.MagicMock()
        visualizer.fit(self.dataset.X)
        visualizer.transform(self.dataset.X)

        # Test that alpha was passed to internal matplotlib scatterplot
        _, scatter_kwargs = visualizer.ax.scatter.call_args
        assert "alpha" in scatter_kwargs
        assert scatter_kwargs["alpha"] == 0.3
        assert pca_array.shape == (self.dataset.X.shape[0], 2)
        
    def test_colorbar(self):
        """
        Test the PCADecomposition visualizer's colorbar features.
        """
        params = {'scale': True, 'proj_dim': 2, 'random_state': 7382, 
                   'color': self.dataset.y, 'colorbar': True}
        visualizer = PCADecomposition(**params).fit(self.dataset.X)
        visualizer.transform(self.dataset.X)
        visualizer.uax.set_xticklabels([])
        # Image comparison tests
        self.assert_images_similar(visualizer)  

    def test_heatmap(self):
        """
        Test the PCADecomposition visualizer's heatmap features.
        """
        params = {'scale': True, 'proj_dim': 2, 'random_state': 7382, 
                   'color': self.dataset.y, 'heatmap': True}
        visualizer = PCADecomposition(**params).fit(self.dataset.X)
        visualizer.transform(self.dataset.X)
        visualizer.finalize()
        # TODO: manually modifying ticks should be removed after #916 is fixed
        visualizer.lax.set_xticks([])
        visualizer.lax.set_yticks([])

        # Image comparison tests
        self.assert_images_similar(visualizer)

    def test_colorbar_heatmap(self):
        """
        Test the PCADecomposition visualizer's colorbar features.
        """
        params = {'scale': True, 'proj_dim': 2, 'random_state': 7382, 
                   'color': self.dataset.y, 'colorbar': True, 'heatmap': True}
        visualizer = PCADecomposition(**params).fit(self.dataset.X)
        visualizer.transform(self.dataset.X)
        visualizer.finalize()
        # TODO: manually modifying ticks should be removed after #916 is fixed
        visualizer.lax.set_xticks([])
        visualizer.uax.set_xticklabels([])
        visualizer.lax.set_yticks([])

        # Image comparison tests
        self.assert_images_similar(visualizer)

    def test_3d_colorbar_heatmap_enabled_error(self):
        """
        Assert an exception if colorbar and heatmap is enabled with 3-dimensions
        """
        with pytest.raises(YellowbrickValueError):
            PCADecomposition(proj_dim=3, colorbar=True)

        with pytest.raises(YellowbrickValueError):
            PCADecomposition(proj_dim=3, heatmap=True)
        

    @pytest.mark.skipif(make_axes_locatable is not None, reason="requires matplotlib <= 2.0.1")
    def test_matplotlib_version_error():
        """
        Assert an exception is raised with incompatible matplotlib versions
        """
        with pytest.raises(YellowbrickValueError):
            PCADecomposition(colorbar=True, heatmap=True)
