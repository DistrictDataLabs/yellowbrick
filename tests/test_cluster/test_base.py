# tests.test_cluster.test_base
# Test the cluster base visualizers.
#
# Author:   Rebecca Bilbro
# Created:  Thu Mar 23 17:38:42 2017 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_base.py [241edca] benjamin@bengfort.com $

"""
Test the cluster base visualizers.
"""

##########################################################################
## Imports
##########################################################################

import pytest

from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.cluster.base import ClusteringScoreVisualizer

from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MeanShift, DBSCAN, Birch
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation


##########################################################################
## Clustering Base Test Cases
##########################################################################


class TestClusterBase(object):
    @pytest.mark.parametrize(
        "model", [SVC, SVR, Ridge, RidgeCV, LinearRegression, RandomForestClassifier]
    )
    def test_clusterer_enforcement_raises(self, model):
        """
        Assert that non-cluster models raise a TypeError for cluster visualizers
        """
        with pytest.raises(YellowbrickTypeError):
            ClusteringScoreVisualizer(model())

    @pytest.mark.parametrize(
        "model",
        [KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, DBSCAN, Birch],
    )
    def test_clusterer_enforcement(self, model):
        """
        Assert that only clustering estimators can be passed to cluster viz
        """
        try:
            ClusteringScoreVisualizer(model())
        except YellowbrickTypeError:
            self.fail("could not pass clustering estimator to visualizer")

    def test_force_estimator(self):
        """
        Test that an estimator can be forced through
        """
        with pytest.raises(YellowbrickTypeError):
            ClusteringScoreVisualizer(LinearRegression())

        try:
            ClusteringScoreVisualizer(LinearRegression(), force_model=True)
        except YellowbrickTypeError as e:
            pytest.fail("type error was raised incorrectly: {}".format(e))
