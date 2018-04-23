# tests.test_features.test_rankd
# Test the rankd feature analysis visualizers
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Oct 07 12:19:19 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_rankd.py [01d5996] benjamin@bengfort.com $

"""
Test the Rankd feature analysis visualizers
"""

##########################################################################
## Imports
##########################################################################

import sys
import six
import pytest

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin, Dataset

from yellowbrick.features.rankd import *
from sklearn.datasets import make_classification

try:
    import pandas as pd
except ImportError:
    pd = None


@pytest.fixture(scope='class')
def dataset(request):
    """
    Creates a binary classification dataset for use in RankD tests
    """
    X, y = make_classification(
        n_samples=700, n_features=10, n_informative=8, n_redundant=2,
        n_classes=2, n_clusters_per_class=2, random_state=6483
    )

    request.cls.dataset = Dataset(X, y)


##########################################################################
## Rank1D Base Tests
##########################################################################

@pytest.mark.usefixtures("dataset")
class TestRank1D(VisualTestCase, DatasetMixin):
    """
    Test the Rank1D visualizer
    """

    def test_rank1d_random(self):
        """
        Test Rank1D on a random binary classification dataset
        """
        visualizer = Rank1D()
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.poof()

        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(pd is None, reason="requires pandas")
    @pytest.mark.filterwarnings("ignore:p-value")
    def test_rank1d_integrated(self):
        """
        Test Rank1D on occupancy dataset with pandas DataFrame and Series
        """
        df = self.load_pandas("occupancy")

        # Load the data from the fixture
        X = df[[
            "temperature", "relative humidity", "light", "C02", "humidity"
        ]]
        y = df['occupancy']

        # Test the visualizer
        visualizer = Rank1D()
        visualizer.fit_transform(X, y)
        visualizer.poof()

        self.assert_images_similar(visualizer)


##########################################################################
## Rank2D Test Cases
##########################################################################

@pytest.mark.usefixtures("dataset")
class TestRank2D(VisualTestCase, DatasetMixin):
    """
    Test the Rank2D visualizer
    """

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_rank2d_random(self):
        """
        Test Rank2D on a random binary classification dataset
        """
        visualizer = Rank2D()
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.poof()

        tol = 10 if six.PY2 else 0.1
        self.assert_images_similar(visualizer, tol=tol)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    @pytest.mark.skipif(pd is None, reason="requires pandas")
    def test_rank2d_integrated(self):
        """
        Test Rank2D on occupancy dataset with pandas DataFrame and Series
        """
        df = self.load_pandas("occupancy")

        # Load the data from the fixture
        X = df[[
            "temperature", "relative humidity", "light", "C02", "humidity"
        ]]
        y = df['occupancy']

        # Test the visualizer
        visualizer = Rank2D()
        visualizer.fit_transform(X, y)
        visualizer.poof()

        tol = 10 if six.PY2 else 0.1
        self.assert_images_similar(visualizer, tol=tol)
#
