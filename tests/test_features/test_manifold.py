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
from yellowbrick.features.base import TargetType
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.datasets.samples_generator import make_s_curve
from sklearn.datasets import make_classification, make_regression, make_blobs

from unittest.mock import patch
from tests.base import VisualTestCase

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Manifold Visualizer Tests
##########################################################################


class TestManifold(VisualTestCase):
    """
    Test Manifold visualizer
    """

    @pytest.mark.parametrize(
        "algorithm",
        ["lle", "ltsa", "hessian", "modified", "isomap", "mds", "spectral", "tsne"],
    )
    def test_manifold_construction(self, algorithm):
        """
        Should be able to construct a manifold estimator from a string
        """
        message = "case failed for {}".format(algorithm)
        params = {"n_neighbors": 18, "random_state": 53}
        oz = Manifold(manifold=algorithm, **params)
        assert is_estimator(oz.manifold), message
        assert oz.manifold.get_params()["n_components"] == 2, message

        manifold_params = oz.manifold.get_params()
        for param, value in params.items():
            if param in manifold_params:
                assert value == manifold_params[param], message

    @pytest.mark.parametrize(
        "algorithm", ["lle", "ltsa", "hessian", "modified", "isomap", "spectral"]
    )
    def test_manifold_warning(self, algorithm):
        """
        Should raise a warning if n_neighbors not specified
        """
        message = "case failed for {}".format(algorithm)
        n_neighbors = 6 if algorithm == "hessian" else 5

        with pytest.warns(YellowbrickWarning):
            oz = Manifold(manifold=algorithm)
            assert oz.n_neighbors == n_neighbors, message

    @pytest.mark.parametrize("algorithm", ["mds", "tsne"])
    def test_manifold_no_warning(self, algorithm):
        """
        Should not raise a warning if n_neighbors not specified
        """
        message = "case failed for {}".format(algorithm)

        with pytest.warns(None) as record:
            assert not record.list, message

    def test_bad_manifold_exception(self):
        """
        Should raise a ValueError when a bad manifold is passed in
        """
        with pytest.raises(YellowbrickValueError, match="could not create manifold"):
            Manifold(manifold=32)

    def test_manifold_instance_construction(self):
        """
        Should allow a sklearn.Estimator object to be set as manifold
        """
        manifold = Pipeline(
            [
                ("pca", PCA(n_components=50)),
                ("lle", LocallyLinearEmbedding(n_components=2)),
            ]
        )

        oz = Manifold(manifold=manifold)
        assert oz.manifold is manifold

    def test_manifold_fit(self):
        """
        Test manifold fit method
        """
        X, y = make_s_curve(1000, random_state=888)
        manifold = Manifold(target="auto")

        assert manifold.fit(X, y) is manifold, "fit did not return self"

    @patch("yellowbrick.features.manifold.Manifold.draw", spec=True)
    def test_manifold_fit_transform(self, mock_draw):
        """
        Test manifold fit_transform method
        """
        X, y = make_s_curve(1000, random_state=888)
        manifold = Manifold(target="auto")

        assert not hasattr(manifold, "fit_time_")

        Xp = manifold.fit_transform(X, y)
        assert Xp.shape == (X.shape[0], 2)

        mock_draw.assert_called_once()
        assert hasattr(manifold, "fit_time_")
        assert manifold._target_color_type == TargetType.CONTINUOUS

    @pytest.mark.filterwarnings("ignore:Conversion of the second argument")
    def test_manifold_classification(self):
        """
        Image similarity test for classification dataset (discrete y)
        """
        X, y = make_classification(
            n_samples=300,
            n_features=7,
            n_informative=4,
            n_redundant=2,
            n_classes=4,
            n_clusters_per_class=2,
            random_state=78,
        )

        oz = Manifold(
            manifold="spectral", target="discrete", n_neighbors=5, random_state=108
        )
        assert not hasattr(oz, "classes_")

        oz.fit_transform(X, y)

        assert hasattr(oz, "classes_")
        assert not hasattr(oz, "range_")
        self.assert_images_similar(oz, tol=0.5)

    def test_manifold_classification_3d(self):
        """
        Image similarity test for classification dataset (discrete y)
        """
        X, y = make_classification(
            n_samples=300,
            n_features=7,
            n_informative=4,
            n_redundant=2,
            n_classes=4,
            n_clusters_per_class=2,
            random_state=78,
        )

        oz = Manifold(
            manifold="spectral",
            target="discrete",
            n_neighbors=5,
            random_state=108,
            projection=3,
        )

        assert not hasattr(oz, "classes_")

        oz.fit_transform(X, y)

        assert hasattr(oz, "classes_")
        assert not hasattr(oz, "range_")
        self.assert_images_similar(oz)

    def test_manifold_regression(self):
        """
        Image similarity test for regression dataset (continuous y)
        """
        X, y = make_regression(
            n_samples=300, n_features=7, n_informative=4, random_state=87
        )

        oz = Manifold(manifold="tsne", target="continuous", random_state=1)
        assert not hasattr(oz, "range_")

        oz.fit_transform(X, y)
        oz.finalize()
        assert not hasattr(oz, "classes_")
        assert hasattr(oz, "range_")
        self.assert_images_similar(oz, tol=1.5)

    def test_manifold_regression_3d(self):
        """
        Image similarity test for regression dataset (continuous y)
        """
        X, y = make_regression(
            n_samples=300, n_features=7, n_informative=4, random_state=87
        )

        oz = Manifold(
            manifold="tsne", target="continuous", random_state=1, projection=3
        )
        assert not hasattr(oz, "range_")

        oz.fit_transform(X, y)
        oz.finalize()
        oz.cbar.set_ticks([])
        assert not hasattr(oz, "classes_")
        assert hasattr(oz, "range_")
        self.assert_images_similar(oz, tol=15)

    def test_manifold_single(self):
        """
        Image similarity test for simple dataset (no y)
        """
        X, _ = make_blobs(n_samples=300, n_features=7, centers=3, random_state=1112)

        oz = Manifold(manifold="mds", random_state=139973)
        oz.fit_transform(X)

        self.assert_images_similar(oz)

    def test_manifold_single_3d(self):
        """
        Image similarity test for simple dataset (no y)
        """
        X, _ = make_blobs(n_samples=300, n_features=7, centers=3, random_state=1112)

        oz = Manifold(manifold="mds", random_state=139973, projection=3)
        oz.fit_transform(X)

        self.assert_images_similar(oz)

    @pytest.mark.skipif(pd is None, reason="requires pandas")
    def test_manifold_pandas(self):
        """
        Test manifold on a dataset made up of a pandas DataFrame and Series
        """
        X, y = make_s_curve(200, random_state=888)
 
        oz = Manifold(
            manifold="ltsa",
            colormap="nipy_spectral",
            n_neighbors=10,
            target="continuous",
            random_state=223,
        )
        oz.fit_transform(X, y)  
        oz.finalize()
        oz.cbar.set_ticks([])
        # TODO: find a way to decrease this tolerance
        self.assert_images_similar(oz, tol=40)

    @pytest.mark.filterwarnings("ignore:Conversion of the second argument")
    @pytest.mark.parametrize(
        "algorithm",
        ["lle", "ltsa", "hessian", "modified", "isomap", "mds", "spectral", "tsne"],
    )
    def test_manifold_algorithm_fit(self, algorithm):
        """
        Test that all algorithms can be fitted correctly
        """
        X, y = make_s_curve(200, random_state=888)
        oz = Manifold(manifold=algorithm, n_neighbors=10, random_state=223)
        oz.fit(X, y)

    def test_manifold_no_transform(self):
        """
        Test the exception when manifold doesn't implement transform.
        """
        X, _ = make_s_curve(1000, random_state=888)
        manifold = Manifold(manifold="mds", target="auto")

        assert not hasattr(manifold._manifold, "transform")

        with pytest.raises(AttributeError, match="try using fit_transform instead"):
            manifold.transform(X)

    @pytest.mark.parametrize("manifolds", ["mds", "spectral", "tsne"])
    def test_manifold_assert_no_transform(self, manifolds):
        """
        Assert that transform raises error when MDS, TSNE or Spectral Embedding algorithms are used.
        """
        X, _ = make_s_curve(1000, random_state=888)
        manifold = Manifold(manifold=manifolds, target="auto", n_neighbors=10)
        manifold.fit(X)
        with pytest.raises(AttributeError, match="try using fit_transform instead"):
            manifold.transform(X)
