# tests.test_features.test_pcoords
# Testing for the parallel coordinates feature visualizers
#
# Author:  Benjamin Bengfort
# Author:  @thekylesaurus
# Created: Thu Oct 06 11:21:27 2016 -0400
#
# Copyright (C) 2017 The scikit-yb developers.
# For license information, see LICENSE.txt
#
# ID: test_pcoords.py [1d407ab] benjamin@bengfort.com $

"""
Testing for the parallel coordinates feature visualizers
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np

from yellowbrick.datasets import load_occupancy
from yellowbrick.features.pcoords import *

from tests.base import VisualTestCase
from ..fixtures import Dataset
from sklearn.datasets import make_classification


try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Fixtures
##########################################################################


@pytest.fixture(scope="class")
def dataset(request):
    """
    Creates a random multiclass classification dataset fixture
    """
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=451,
        flip_y=0,
        class_sep=3,
        scale=np.array([1.0, 2.0, 100.0, 20.0, 1.0]),
    )

    dataset = Dataset(X, y)
    request.cls.dataset = dataset


##########################################################################
## Parallel Coordinates Tests
##########################################################################


@pytest.mark.usefixtures("dataset")
class TestParallelCoordinates(VisualTestCase):
    """
    Test the ParallelCoordinates visualizer
    """

    def test_parallel_coords(self):
        """
        Test images closeness on random 3 class dataset
        """
        visualizer = ParallelCoordinates()
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_parallel_coords_fast(self):
        """
        Test images closeness on random 3 class dataset in fast mode
        """
        visualizer = ParallelCoordinates(fast=True)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_alpha(self):
        """
        Test image closeness on opaque alpha for random 3 class dataset
        """
        visualizer = ParallelCoordinates(alpha=1.0)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_alpha_fast(self):
        """
        Test image closeness on opaque alpha for random 3 class dataset in fast mode
        """
        visualizer = ParallelCoordinates(alpha=1.0, fast=True)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_labels(self):
        """
        Test image closeness when class and feature labels are supplied
        """
        visualizer = ParallelCoordinates(
            classes=["a", "b", "c"], features=["f1", "f2", "f3", "f4", "f5"]
        )
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer)

    def test_labels_fast(self):
        """
        Test image closeness when class and feature labels are supplied in fast mode
        """
        visualizer = ParallelCoordinates(
            classes=["a", "b", "c"], features=["f1", "f2", "f3", "f4", "f5"], fast=True
        )
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer)

    def test_normalized_l2(self):
        """
        Test image closeness on l2 normalized 3 class dataset
        """
        visualizer = ParallelCoordinates(normalize="l2")
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_normalized_l2_fast(self):
        """
        Test image closeness on l2 normalized 3 class dataset in fast mode
        """
        visualizer = ParallelCoordinates(normalize="l2", fast=True)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_normalized_minmax(self):
        """
        Test image closeness on minmax normalized 3 class dataset
        """
        visualizer = ParallelCoordinates(normalize="minmax")
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_normalized_minmax_fast(self):
        """
        Test image closeness on minmax normalized 3 class dataset in fast mode
        """
        visualizer = ParallelCoordinates(normalize="minmax", fast=True)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_parallel_coordinates_quickmethod(self):
        """
        Test the quick method producing a valid visualization
        """
        X, y = load_occupancy(return_dataset=True).to_numpy()

        # Compare the images
        # Use only the first 100 samples so the test will run faster
        visualizer = parallel_coordinates(X, y, sample=100, show=False)
        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_integration_sampled(self):
        """
        Test on a real dataset with pandas DataFrame and Series sampled for speed
        """
        data = load_occupancy(return_dataset=True)
        X, y = data.to_pandas()
        classes = [
            k for k, _ in sorted(data.meta["labels"].items(), key=lambda i: i[1])
        ]

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        oz = ParallelCoordinates(
            sample=0.05, shuffle=True, random_state=4291, classes=classes
        )
        oz.fit_transform(X, y)
        oz.finalize()

        self.assert_images_similar(oz, tol=0.1)

    def test_numpy_integration_sampled(self):
        """
        Ensure visualizer works in default case with numpy arrays and sampling
        """
        data = load_occupancy(return_dataset=True)
        X, y = data.to_numpy()
        classes = [
            k for k, _ in sorted(data.meta["labels"].items(), key=lambda i: i[1])
        ]

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

        oz = ParallelCoordinates(
            sample=0.05, shuffle=True, random_state=4291, classes=classes
        )
        oz.fit_transform(X, y)
        oz.finalize()

        self.assert_images_similar(oz, tol=0.1)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_integration_fast(self):
        """
        Test on a real dataset with pandas DataFrame and Series in fast mode
        """
        data = load_occupancy(return_dataset=True)
        X, y = data.to_pandas()
        classes = [
            k for k, _ in sorted(data.meta["labels"].items(), key=lambda i: i[1])
        ]

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

        oz = ParallelCoordinates(fast=True, classes=classes)
        oz.fit_transform(X, y)
        oz.finalize()

        self.assert_images_similar(oz, tol=0.1)

    def test_numpy_integration_fast(self):
        """
        Ensure visualizer works in default case with numpy arrays and fast mode
        """
        data = load_occupancy(return_dataset=True)
        X, y = data.to_numpy()
        classes = [
            k for k, _ in sorted(data.meta["labels"].items(), key=lambda i: i[1])
        ]

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

        oz = ParallelCoordinates(fast=True, classes=classes)
        oz.fit_transform(X, y)
        oz.finalize()

        self.assert_images_similar(oz, tol=0.1)

    def test_normalized_invalid_arg(self):
        """
        Invalid argument to 'normalize' should raise
        """
        with pytest.raises(YellowbrickValueError):
            ParallelCoordinates(normalize="foo")

    def test_sample_int(self):
        """
        Assert no errors occur using integer 'sample' argument
        """
        visualizer = ParallelCoordinates(sample=10)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

    def test_sample_int_shuffle(self):
        """
        Assert no errors occur using integer 'sample' argument and shuffle, with different random_state args
        """
        visualizer = ParallelCoordinates(sample=3, shuffle=True)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

        visualizer = ParallelCoordinates(sample=3, shuffle=True, random_state=444)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

        visualizer = ParallelCoordinates(
            sample=3, shuffle=True, random_state=np.random.RandomState()
        )
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

    def test_sample_int_shuffle_false(self):
        """
        Assert no errors occur using integer 'sample' argument and shuffle, with different random_state args
        """
        visualizer = ParallelCoordinates(sample=3, shuffle=False)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

        visualizer = ParallelCoordinates(sample=3, shuffle=False, random_state=444)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

        visualizer = ParallelCoordinates(
            sample=3, shuffle=False, random_state=np.random.RandomState()
        )
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

    def test_sample_int_invalid(self):
        """
        Negative int values should raise exception
        """
        with pytest.raises(YellowbrickValueError):
            ParallelCoordinates(sample=-1)

    def test_sample_float(self):
        """
        Assert no errors occur using float 'sample' argument
        """
        visualizer = ParallelCoordinates(sample=0.5)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

    def test_sample_float_shuffle(self):
        """
        Assert no errors occur using float 'sample' argument and shuffle, with different random_state args
        """
        visualizer = ParallelCoordinates(sample=0.5, shuffle=True)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

        visualizer = ParallelCoordinates(sample=0.5, shuffle=True, random_state=444)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

        visualizer = ParallelCoordinates(
            sample=0.5, shuffle=True, random_state=np.random.RandomState()
        )
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

    def test_sample_float_shuffle_false(self):
        """
        Assert no errors occur using float 'sample' argument and shuffle, with different random_state args
        """
        visualizer = ParallelCoordinates(sample=0.5, shuffle=False)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

        visualizer = ParallelCoordinates(sample=0.5, shuffle=False, random_state=444)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

        visualizer = ParallelCoordinates(
            sample=0.5, shuffle=False, random_state=np.random.RandomState()
        )
        visualizer.fit_transform(self.dataset.X, self.dataset.y)

    def test_sample_float_invalid(self):
        """
        Float values for 'sample' argument outside [0,1] should raise.
        """
        with pytest.raises(YellowbrickValueError):
            ParallelCoordinates(sample=-0.2)

        with pytest.raises(YellowbrickValueError):
            ParallelCoordinates(sample=1.1)

    def test_sample_invalid_type(self):
        """
        Non-numeric values for 'sample' argument should raise.
        """
        with pytest.raises(YellowbrickTypeError):
            ParallelCoordinates(sample="foo")

    @staticmethod
    def test_static_subsample():
        """
        Assert output of subsampling method against expectations
        """

        ntotal = 100
        ncols = 50

        y = np.arange(ntotal)
        X = np.ones((ntotal, ncols)) * y.reshape(ntotal, 1)

        visualizer = ParallelCoordinates(sample=1.0, random_state=None, shuffle=False)
        Xprime, yprime = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X)
        assert np.array_equal(yprime, y)

        visualizer = ParallelCoordinates(sample=200, random_state=None, shuffle=False)
        Xprime, yprime = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X)
        assert np.array_equal(yprime, y)

        sample = 50
        visualizer = ParallelCoordinates(
            sample=sample, random_state=None, shuffle=False
        )
        Xprime, yprime = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X[:sample, :])
        assert np.array_equal(yprime, y[:sample])

        sample = 50
        visualizer = ParallelCoordinates(sample=sample, random_state=None, shuffle=True)
        Xprime, yprime = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X[yprime.flatten(), :])
        assert len(Xprime) == sample
        assert len(yprime) == sample

        visualizer = ParallelCoordinates(sample=0.5, random_state=None, shuffle=False)
        Xprime, yprime = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X[: int(ntotal / 2), :])
        assert np.array_equal(yprime, y[: int(ntotal / 2)])

        sample = 0.5
        visualizer = ParallelCoordinates(sample=sample, random_state=None, shuffle=True)
        Xprime, yprime = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X[yprime.flatten(), :])
        assert len(Xprime) == ntotal * sample
        assert len(yprime) == ntotal * sample

        sample = 0.25
        visualizer = ParallelCoordinates(sample=sample, random_state=444, shuffle=True)
        Xprime, yprime = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X[yprime.flatten(), :])
        assert len(Xprime) == ntotal * sample
        assert len(yprime) == ntotal * sample

        sample = 0.99
        visualizer = ParallelCoordinates(
            sample=sample, random_state=np.random.RandomState(), shuffle=True
        )
        Xprime, yprime = visualizer._subsample(X, y)
        assert np.array_equal(Xprime, X[yprime.flatten(), :])
        assert len(Xprime) == ntotal * sample
        assert len(yprime) == ntotal * sample
