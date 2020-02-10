# tests.test_features.test_manifold
# Tests for the Manifold High Dimensional Visualizations
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Sat May 12 11:24:41 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_manifold.py [02f8c27] benjamin@bengfort.com $

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
from yellowbrick.exceptions import YellowbrickValueError, ModelError, NotFitted

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.datasets import make_s_curve
from sklearn.manifold import LocallyLinearEmbedding

from unittest.mock import patch
from tests.base import VisualTestCase

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Manifold Visualizer Tests
##########################################################################


@pytest.mark.usefixtures("s_curves", "discrete", "continuous")
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

    @pytest.mark.filterwarnings("ignore:Conversion of the second argument")
    @pytest.mark.parametrize(
        "algorithm", ["lle", "ltsa", "hessian", "modified", "isomap"]
    )
    def test_manifold_algorithm_transform_fit(self, algorithm):
        """
        Test manifold fit with algorithms having transform implemented
        """
        X, y = make_s_curve(1000, random_state=94)
        with pytest.warns(YellowbrickWarning):
            manifold = Manifold(manifold=algorithm, target="auto")

        assert manifold.fit(X, y) is manifold, "fit did not return self"

    @pytest.mark.filterwarnings("ignore:Conversion of the second argument")
    @pytest.mark.parametrize("algorithm", ["mds", "spectral", "tsne"])
    def test_manifold_algorithm_no_transform_fit(self, algorithm):
        """
        Test manifold fit with algorithms not having transform implemented
        """
        X, y = self.s_curves
        msg = "requires data to be simultaneously fit and transformed"
        oz = Manifold(manifold=algorithm, n_neighbors=10, random_state=223)
        with pytest.raises(ModelError, match=msg):
            oz.fit(X)

    @patch("yellowbrick.features.manifold.Manifold.draw", spec=True)
    @pytest.mark.parametrize("projection", [2, 3])
    def test_manifold_fit_transform(self, mock_draw, projection):
        """
        Test manifold fit_transform method
        """
        X, y = self.s_curves
        manifold = Manifold(target="auto", projection=projection)

        assert not hasattr(manifold, "fit_time_")

        Xp = manifold.fit_transform(X, y)
        assert Xp.shape == (X.shape[0], projection)

        mock_draw.assert_called_once()
        assert hasattr(manifold, "fit_time_")
        assert manifold._target_color_type == TargetType.CONTINUOUS

    @patch("yellowbrick.features.manifold.Manifold.fit_transform", spec=True)
    @patch("yellowbrick.features.manifold.Manifold.draw", spec=True)
    @pytest.mark.parametrize("projection", [2, 3])
    def test_manifold_transform(self, mock_draw, mock_fit_transform, projection):
        """
        Test manifold transform method
        """
        X, y = self.s_curves
        manifold = Manifold(
            manifold="lle", target="auto", n_neighbors=5, projection=projection
        )

        manifold.fit(X, y)
        Xp = manifold.transform(X, y)
        assert Xp.shape == (X.shape[0], projection)

        mock_draw.assert_called_once()

    def test_manifold_no_transform(self):
        """
        Test the exception when manifold doesn't implement transform.
        """
        X, _ = self.s_curves
        manifold = Manifold(manifold="lle", n_neighbors=5, target="auto")

        msg = "instance is not fitted yet, please call fit"
        with pytest.raises(NotFitted, match=msg):
            manifold.transform(X)

    @patch("yellowbrick.features.manifold.Manifold.fit", spec=True)
    @pytest.mark.parametrize("manifolds", ["mds", "spectral", "tsne"])
    def test_manifold_assert_no_transform(self, mock_fit, manifolds):
        """
        Assert that transform raises error when MDS, TSNE or Spectral Embedding algorithms are used.
        """
        X, _ = self.s_curves
        manifold = Manifold(manifold=manifolds, target="auto", n_neighbors=10)
        mock_fit(X)
        msg = "requires data to be simultaneously fit and transformed"
        with pytest.raises(ModelError, match=msg):
            manifold.transform(X)

    @pytest.mark.filterwarnings("ignore:Conversion of the second argument")
    def test_manifold_classification(self):
        """
        Image similarity test for classification dataset (discrete y)
        """
        X, y = self.discrete

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
        X, y = self.discrete

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
        X, y = self.continuous

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
        X, y = self.continuous

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
        X, y = self.s_curves

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

    def test_manifold_quick_method_no_target(self):
        """
        Test Manifold quick method with no target.
        """
        X, _ = make_blobs(n_samples=300, n_features=7, centers=3, random_state=1112)
        visualizer = manifold_embedding(
            X, manifold="mds", random_state=139973, show=False
        )

        assert isinstance(visualizer, Manifold)
        self.assert_images_similar(visualizer)

    def test_manifold_quick_method_discrete_target(self):
        """
        Test Manifold quick method with a discrete target.
        """
        X, y = self.discrete

        visualizer = manifold_embedding(
            X,
            y,
            manifold="mds",
            target="discrete",
            n_neighbors=5,
            random_state=37,
            show=False
        )
        assert isinstance(visualizer, Manifold)
        self.assert_images_similar(visualizer)

    def test_manifold_quick_method_continuous_target(self):
        """
        Test Manifold quick method with a continuous target.
        """
        X, y = self.continuous

        visualizer = manifold_embedding(
            X,
            y,
            manifold="tsne",
            target="continuous",
            random_state=37,
            show=False
        )
        assert isinstance(visualizer, Manifold)

        # ImageComparisonFailure: images not close (RMS 1.124) on Miniconda
        self.assert_images_similar(visualizer, tol=1.5)
