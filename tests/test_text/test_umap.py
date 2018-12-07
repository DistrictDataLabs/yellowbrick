# tests.test_text.test_umap
# Tests for the UMAP visual corpus embedding mechanism.
#
# Author:   John Healy <jchealy@gmail.com>
# Created:  Mon Dec 03, 14:00:00
#
# Copyright (C) 2016 Bengfort.com
# For license information, see LICENSE.txt
#
#

"""
Tests for the UMAP visual corpus embedding mechanism.
"""

##########################################################################
## Imports
##########################################################################

import six
import pytest

from yellowbrick.text.umap_vis import *
from tests.base import VisualTestCase
from tests.dataset import DatasetMixin
from yellowbrick.exceptions import YellowbrickValueError

from umap import UMAP
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import pandas
except ImportError:
    pandas = None

try:
    from unittest import mock
except ImportError:
    import mock

##########################################################################
## TSNE Tests
##########################################################################

class TestUMAP(VisualTestCase, DatasetMixin):
    """
    UMAPVisualizer tests
    """

    def test_make_pipeline(self):
        """
        Verify the pipeline creation step for UMAP
        """

        umap = UMAPVisualizer() # Should not cause an exception.
        assert umap.transformer_ is not None

        assert len(umap.transformer_.steps) == 1

    def test_integrated_umap(self):
        """
        Check UMAP integrated visualization on the hobbies corpus
        """
        corpus = self.load_data('hobbies')
        tfidf  = TfidfVectorizer()

        docs   = tfidf.fit_transform(corpus.data)
        labels = corpus.target

        umap = UMAPVisualizer(random_state=8392, colormap='Set1', alpha=1.0)
        umap.fit_transform(docs, labels)

        tol = 50 if six.PY3 else 55
        self.assert_images_similar(umap, tol=tol)

    def test_sklearn_umap_size(self):
        """
        Check to make sure sklearn's UMAP doesn't use the size param
        """
        # In UMAPVisualizer, the internal sklearn UMAP transform consumes
        # some but not all kwargs passed in by user. Those not in get_params(),
        # like size, are passed through to YB's finalize method. This test should
        # notify us  if UMAP's params change on the sklearn side.
        with pytest.raises(TypeError):
            UMAP(size=(100,100))

    def test_sklearn_umap_title(self):
        """
        Check to make sure sklearn's UMAP doesn't use the title param
        """
        # In TSNEVisualizer, the internal sklearn UMAP transform consumes
        # some but not all kwargs passed in by user. Those not in get_params(),
        # like title, are passed through to YB's finalize method. This test should
        # notify us  if UMAP's params change on the sklearn side.
        with pytest.raises(TypeError):
            UMAP(title="custom_title")

    def test_custom_title_umap(self):
        """
        Check UMAP can accept a custom title (string) from the user
        """
        umap = UMAPVisualizer(title="custom_title")

        assert umap.title == "custom_title"

    def test_custom_size_umap(self):
        """
        Check UMAP can accept a custom size (tuple of pixels) from the user
        """
        umap = UMAPVisualizer(size=(100, 50))

        assert umap._size == (100, 50)

    def test_make_classification_umap(self):
        """
        Test UMAP integrated visualization on a sklearn classifier dataset
        """

        ## produce random data
        X, y = make_classification(n_samples=200, n_features=100,
                               n_informative=20, n_redundant=10,
                               n_classes=3, random_state=42)

        ## visualize data with UMAP
        umap = UMAPVisualizer(random_state=87)
        umap.fit(X, y)

        tol = 0.1 if six.PY3 else 40
        self.assert_images_similar(umap, tol=tol)

    def test_make_classification_umap_class_labels(self):
        """
        Test UMAP integrated visualization with class labels specified
        """

        ## produce random data
        X, y = make_classification(n_samples=200, n_features=100,
                               n_informative=20, n_redundant=10,
                               n_classes=3, random_state=42)

        ## visualize data with UMAP
        umap = UMAPVisualizer(random_state=87, labels=['a', 'b', 'c'])
        umap.fit(X, y)

        tol = 0.1 if six.PY3 else 40
        self.assert_images_similar(umap, tol=tol)

    def test_umap_mismtached_labels(self):
        """
        Assert exception is raised when number of labels doesn't match
        """
        ## produce random data
        X, y = make_classification(n_samples=200, n_features=100,
                               n_informative=20, n_redundant=10,
                               n_classes=3, random_state=42)

        ## fewer labels than classes
        umap = UMAPVisualizer(random_state=87, labels=['a', 'b'])
        with pytest.raises(YellowbrickValueError):
            umap.fit(X,y)

        ## more labels than classes
        umap = UMAPVisualizer(random_state=87, labels=['a', 'b', 'c', 'd'])
        with pytest.raises(YellowbrickValueError):
            umap.fit(X,y)

    def test_no_target_umap(self):
        """
        Test UMAP when no target or classes are specified
        """
        ## produce random data
        X, y = make_classification(n_samples=200, n_features=100,
                               n_informative=20, n_redundant=10,
                               n_classes=3, random_state=6897)

        ## visualize data with UMAP
        umap = UMAPVisualizer(random_state=64)
        umap.fit(X)

        self.assert_images_similar(umap, tol=0.1)

    @pytest.mark.skipif(pandas is None, reason="test requires pandas")
    def test_visualizer_with_pandas(self):
        """
        Test UMAP when passed a pandas DataFrame and series
        """
        X, y = make_classification(
            n_samples=200, n_features=100, n_informative=20, n_redundant=10,
            n_classes=3, random_state=3020
        )

        X = pandas.DataFrame(X)
        y = pandas.Series(y)

        umap = UMAPVisualizer(random_state=64)
        umap.fit(X, y)

        tol = 0.1 if six.PY3 else 40
        self.assert_images_similar(umap, tol=tol)

    def test_alpha_param(self):
        """
        Test that the user can supply an alpha param on instantiation
        """
        ## produce random data
        X, y = make_classification(n_samples=200, n_features=100,
                               n_informative=20, n_redundant=10,
                               n_classes=3, random_state=42)

        ## Instantiate a UMAPVisualizer, provide custom alpha
        umap = UMAPVisualizer(random_state=64, alpha=0.5)

        # Test param gets set correctly
        assert umap.alpha == 0.5

        # Mock ax and fit the visualizer
        umap.ax = mock.MagicMock(autospec=True)
        umap.fit(X, y)

        # Test that alpha was passed to internal matplotlib scatterplot
        _, scatter_kwargs = umap.ax.scatter.call_args
        assert "alpha" in scatter_kwargs
        assert scatter_kwargs["alpha"] == 0.5
