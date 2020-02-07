# tests.test_text.test_umap
# Tests for the UMAP visual corpus embedding mechanism.
#
# Author:   John Healy
# Created:  Mon Dec 03, 14:00:00
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_umap.py [] jchealy@gmail.com> $

"""
Tests for the UMAP visual corpus embedding mechanism.
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest
import warnings

from unittest import mock
from tests.base import VisualTestCase
from yellowbrick.text.umap_vis import *
from yellowbrick.datasets import load_hobbies
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import pandas
except ImportError:
    pandas = None

try:
    from umap import UMAP
except ImportError:
    UMAP = None
except (RuntimeError, AttributeError):
    UMAP = None
    warnings.warn(
        "Error Importing UMAP.  UMAP does not support python 2.7 on Windows 32 bit."
    )


##########################################################################
## Data
##########################################################################

corpus = load_hobbies()


##########################################################################
## UMAP Tests
##########################################################################


@mock.patch("yellowbrick.text.umap_vis.UMAP", None)
def test_umap_unavailable():
    """
    Assert an appropriate exception is raised when UMAP is not installed
    """
    from yellowbrick.text.umap_vis import UMAP

    assert UMAP is None

    with pytest.raises(
        YellowbrickValueError, match="umap package doesn't seem to be installed"
    ):
        UMAPVisualizer()


@pytest.mark.skipif(UMAP is None, reason="tests require the umap library")
@pytest.mark.xfail(
    sys.platform == "win32", reason="not supported on windows 32bit with Python 2.7"
)
class TestUMAP(VisualTestCase):
    """
    UMAPVisualizer tests
    """

    def test_make_pipeline(self):
        """
        Verify the pipeline creation step for UMAP
        """

        umap = UMAPVisualizer()  # Should not cause an exception.
        assert umap.transformer_ is not None

        assert len(umap.transformer_.steps) == 1

    def test_integrated_umap(self):
        """
        Check UMAP integrated visualization on the hobbies corpus
        """
        tfidf = TfidfVectorizer()

        docs = tfidf.fit_transform(corpus.data)
        labels = corpus.target

        umap = UMAPVisualizer(random_state=8392, colormap="Set1", alpha=1.0)
        umap.fit_transform(docs, labels)

        tol = 55
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
            UMAP(size=(100, 100))

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

    def test_custom_colors_umap(self):
        """
        Check UMAP accepts and properly handles custom colors from user
        """
        ## produce random data
        X, y = make_classification(
            n_samples=200,
            n_features=100,
            n_informative=20,
            n_redundant=10,
            n_classes=5,
            random_state=42,
        )

        ## specify a list of custom colors >= n_classes
        purple_blues = ["indigo", "orchid", "plum", "navy", "purple", "blue"]

        ## instantiate the visualizer and check that self.colors is correct
        purple_umap = UMAPVisualizer(colors=purple_blues, random_state=87)
        assert purple_umap.colors == purple_blues

        ## fit the visualizer and check that self.color_values is as long as
        ## n_classes and is the first n_classes items in self.colors
        purple_umap.fit(X, y)
        assert len(purple_umap.color_values_) == len(purple_umap.classes_)
        assert purple_umap.color_values_ == purple_blues[: len(purple_umap.classes_)]

        ## specify a list of custom colors < n_classes
        greens = ["green", "lime", "teal"]

        ## instantiate the visualizer and check that self.colors is correct
        green_umap = UMAPVisualizer(colors=greens, random_state=87)
        assert green_umap.colors == greens

        ## fit the visualizer and check that self.color_values is as long as
        ## n_classes and the user-supplied color list gets recycled as expected
        green_umap.fit(X, y)
        assert len(green_umap.color_values_) == len(green_umap.classes_)
        assert green_umap.color_values_ == ["green", "lime", "teal", "green", "lime"]

    def test_make_classification_umap(self):
        """
        Test UMAP integrated visualization on a sklearn classifier dataset
        """

        ## produce random data
        X, y = make_classification(
            n_samples=200,
            n_features=100,
            n_informative=20,
            n_redundant=10,
            n_classes=3,
            random_state=42,
        )

        ## visualize data with UMAP
        umap = UMAPVisualizer(random_state=87)
        umap.fit(X, y)

        self.assert_images_similar(umap, tol=40)

    def test_make_classification_umap_class_labels(self):
        """
        Test UMAP integrated visualization with class labels specified
        """

        ## produce random data
        X, y = make_classification(
            n_samples=200,
            n_features=100,
            n_informative=20,
            n_redundant=10,
            n_classes=3,
            random_state=42,
        )

        ## visualize data with UMAP
        umap = UMAPVisualizer(random_state=87, labels=["a", "b", "c"])
        umap.fit(X, y)

        self.assert_images_similar(umap, tol=40)

    def test_umap_mismtached_labels(self):
        """
        Assert exception is raised when number of labels doesn't match
        """
        ## produce random data
        X, y = make_classification(
            n_samples=200,
            n_features=100,
            n_informative=20,
            n_redundant=10,
            n_classes=3,
            random_state=42,
        )

        ## fewer labels than classes
        umap = UMAPVisualizer(random_state=87, labels=["a", "b"])
        with pytest.raises(YellowbrickValueError):
            umap.fit(X, y)

        ## more labels than classes
        umap = UMAPVisualizer(random_state=87, labels=["a", "b", "c", "d"])
        with pytest.raises(YellowbrickValueError):
            umap.fit(X, y)

    def test_no_target_umap(self):
        """
        Test UMAP when no target or classes are specified
        """
        ## produce random data
        X, y = make_classification(
            n_samples=200,
            n_features=100,
            n_informative=20,
            n_redundant=10,
            n_classes=3,
            random_state=6897,
        )

        ## visualize data with UMAP
        umap = UMAPVisualizer(random_state=64)
        umap.fit(X)

        self.assert_images_similar(umap, tol=40)

    @pytest.mark.skipif(pandas is None, reason="test requires pandas")
    def test_visualizer_with_pandas(self):
        """
        Test UMAP when passed a pandas DataFrame and series
        """
        X, y = make_classification(
            n_samples=200,
            n_features=100,
            n_informative=20,
            n_redundant=10,
            n_classes=3,
            random_state=3020,
        )

        X = pandas.DataFrame(X)
        y = pandas.Series(y)

        umap = UMAPVisualizer(random_state=64)
        umap.fit(X, y)

        self.assert_images_similar(umap, tol=40)

    def test_alpha_param(self):
        """
        Test that the user can supply an alpha param on instantiation
        """
        ## produce random data
        X, y = make_classification(
            n_samples=200,
            n_features=100,
            n_informative=20,
            n_redundant=10,
            n_classes=3,
            random_state=42,
        )

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

    def test_quick_method(self):
        """
        Test for umap quick  method with hobbies dataset
        """
        corpus = load_hobbies()
        tfidf = TfidfVectorizer()

        X = tfidf.fit_transform(corpus.data)
        y = corpus.target

        viz = umap(X, y, show=False)
        assert isinstance(viz, UMAPVisualizer)

        self.assert_images_similar(viz, tol=50)

