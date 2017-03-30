# tests.test_cluster.test_elbow
# Tests for the KElbowVisualizer
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Mar 23 22:30:19 2017 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_elbow.py [] benjamin@bengfort.com $

"""
Tests for the KElbowVisualizer
"""

##########################################################################
## Imports
##########################################################################

from ..base import VisualTestCase
from ..dataset import DatasetMixin

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
from yellowbrick.cluster.elbow import KElbowVisualizer
from yellowbrick.exceptions import YellowbrickValueError


##########################################################################
## KElbowVisualizer Test Cases
##########################################################################

class KElbowVisualizerTests(VisualTestCase, DatasetMixin):

    def test_integrated_kmeans_elbow(self):
        """
        Test no exceptions for kmeans k-elbow visualizer on blobs dataset

        See #182: cannot use occupancy dataset because of memory usage
        """

        # Generate a blobs data set
        X,y = make_blobs(
            n_samples=1000, n_features=12, centers=6, shuffle=True
        )

        try:
            visualizer = KElbowVisualizer(KMeans(), k=4)
            visualizer.fit(X)
            visualizer.poof()
        except Exception as e:
            self.fail("error during k-elbow: {}".format(e))

    def test_integrated_mini_batch_kmeans_elbow(self):
        """
        Test no exceptions for mini-batch kmeans k-elbow visualizer

        See #182: cannot use occupancy dataset because of memory usage
        """

        # Generate a blobs data set
        X,y = make_blobs(
            n_samples=1000, n_features=12, centers=6, shuffle=True
        )

        try:
            visualizer = KElbowVisualizer(MiniBatchKMeans(), k=4)
            visualizer.fit(X)
            visualizer.poof()
        except Exception as e:
            self.fail("error during k-elbow: {}".format(e))

    def test_invalid_k(self):
        """
        Assert that invalid values of K raise exceptions
        """

        with self.assertRaises(YellowbrickValueError):
            model = KElbowVisualizer(KMeans(), k=(1,2,3,4,5))

        with self.assertRaises(YellowbrickValueError):
            model = KElbowVisualizer(KMeans(), k="foo")
