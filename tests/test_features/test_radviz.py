# tests.test_features.test_radviz
# Test the RadViz feature analysis visualizers
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Oct 07 12:19:19 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_radviz.py [01d5996] benjamin@bengfort.com $

"""
Test the RadViz feature analysis visualizers
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest
import numpy.testing as npt

from tests.base import VisualTestCase
from ..fixtures import TestDataset
from sklearn.datasets import make_classification

from yellowbrick.datasets import load_occupancy
from yellowbrick.features.radviz import *

try:
    import pandas
except ImportError:
    pandas = None


##########################################################################
## Fixtures
##########################################################################

@pytest.fixture(scope='class')
def dataset(request):
    """
    Creates a random multiclass classification dataset fixture
    """
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=451, flip_y=0,
        class_sep=3, scale=np.array([1.0, 2.0, 100.0, 20.0, 1.0])
    )

    dataset = TestDataset(X, y)
    request.cls.dataset = dataset


##########################################################################
## RadViz Tests
##########################################################################

@pytest.mark.usefixtures('dataset')
class TestRadViz(VisualTestCase):
    """
    Test the RadViz visualizer
    """

    def test_normalize_x(self):
        """
        Test the static normalization method on the RadViz class
        """
        # Original data
        X = np.array(
            [[ 2.318, 2.727, 4.260, 7.212, 4.792],
             [ 2.315, 2.726, 4.295, 7.140, 4.783,],
             [ 2.315, 2.724, 4.260, 7.135, 4.779,],
             [ 2.110, 3.609, 4.330, 7.985, 5.595,],
             [ 2.110, 3.626, 4.330, 8.203, 5.621,],
             [ 2.110, 3.620, 4.470, 8.210, 5.612,]]
        )

        # Expected result
        Xe = np.array(
            [[ 1.        ,  0.00332594,  0.        ,  0.07162791,  0.01543943],
             [ 0.98557692,  0.00221729,  0.16666667,  0.00465116,  0.00475059],
             [ 0.98557692,  0.        ,  0.        ,  0.        ,  0.        ],
             [ 0.        ,  0.98115299,  0.33333333,  0.79069767,  0.96912114],
             [ 0.        ,  1.        ,  0.33333333,  0.99348837,  1.        ],
             [ 0.        ,  0.99334812,  1.        ,  1.        ,  0.98931116]]
        )

        # Xprime (transformed X)
        Xp = RadViz.normalize(X)
        npt.assert_array_almost_equal(Xp, Xe)

    def test_radviz(self):
        """
        Assert image similarity on test dataset
        """
        visualizer = RadViz()
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_radviz_alpha(self):
        """
        Assert image similarity with alpha transparency
        """
        visualizer = RadViz(alpha=0.5)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=0.25)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_integrated_radiz_with_pandas(self):
        """
        Test RadViz with Pandas on the occupancy dataset
        """
        data = load_occupancy(return_dataset=True)
        X, y = data.to_data()

        if pandas is None:
            features = data.meta["features"]
        else:
            assert isinstance(X, pandas.DataFrame)
            assert isinstance(y, pandas.Series)

            npt.assert_equal(X.columns.values, data.meta["features"])

            features = None

        # Test the visualizer
        visualizer = RadViz(features=features)
        visualizer.fit_transform_poof(X, y)
        self.assert_images_similar(visualizer, tol=0.1)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_integrated_radiz_pandas_classes_features(self):
        """
        Test RadViz with classes and features specified
        """
        # Load the data from the fixture
        data = load_occupancy(return_dataset=True)
        X, y = data.to_data()

        features = ["temperature", "relative humidity", "light"]
        classes = ['unoccupied', 'occupied']

        if pandas is None:
            X = X[:, :3]
        else:
            X = X[features]
            y = y.astype(int)

            assert isinstance(X, pandas.DataFrame)
            assert isinstance(y, pandas.Series)

        # Test the visualizer
        visualizer = RadViz(features=features, classes=classes)
        visualizer.fit_transform_poof(X, y)
        self.assert_images_similar(visualizer, tol=0.1)
