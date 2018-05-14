# tests.test_features.test_manifold
# Tests for the Manifold High Dimensional Visualizations
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Sat May 12 11:24:41 2018 -0400
#
# ID: test_manifold.py [] benjamin@bengfort.com $

"""
Tests for the Manifold High Dimensional Visualizations
"""

##########################################################################
## Imports
##########################################################################

import pytest

from yellowbrick.features.manifold import *
from yellowbrick.utils.types import is_estimator
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.datasets.samples_generator import make_s_curve

from tests.base import VisualTestCase


##########################################################################
## Manifold Visualizer Tests
##########################################################################

class TestManifold(VisualTestCase):
    """
    Test Manifold visualizer
    """

    def test_manifold_construction(self):
        """
        Should be able to construct a manifold estimator from a string
        """
        # TODO: parametrize this once unittest.TestCase dependency removed.
        algorithms = [
            "lle", "ltsa", "hessian", "modified",
            "isomap", "mds", "spectral", "tsne",
        ]

        for algorithm in algorithms:
            message = "case failed for {}".format(algorithm)
            params = {
                "n_neighbors": 18,
                "random_state": 53,
            }
            oz = Manifold(manifold=algorithm, **params)
            assert is_estimator(oz.manifold), message
            assert oz.manifold.get_params()["n_components"] == 2, message

            manifold_params = oz.manifold.get_params()
            for param, value in params.items():
                if param in manifold_params:
                    assert value == manifold_params[param], message

    def test_bad_manifold_exception(self):
        """
        Should raise a ValueError when a bad manifest is passed in
        """
        with pytest.raises(YellowbrickValueError, match="could not create manifold"):
            Manifold(manifold=32)

    def test_manifold_instance_construction(self):
        """
        Should allow a sklearn.Estimator object to be set as manifold
        """
        manifold = Pipeline([
            ('pca', PCA(n_components=50)),
            ('lle', LocallyLinearEmbedding(n_components=2)),
        ])

        oz = Manifold(manifold=manifold)
        assert oz.manifold is manifold

    @pytest.mark.skip(reason="requires parametrize")
    def test_manifold_image_similarity(self):
        """
        Perform image similarity test on default manifold dataset
        """
        # TODO: parametrize this once unittest.TestCase dependency removed.
        algorithms = [
            "lle", "ltsa", "hessian", "modified",
            "isomap", "mds", "spectral", "tsne",
        ]

        X, y = make_s_curve(1000, random_state=888)

        for algorithm in algorithms:
            oz = Manifold(manifold=algorithm, random_state=223)
            oz.fit(X, y)
            self.assert_images_similar(oz, tol=1.0)
