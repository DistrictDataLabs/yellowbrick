# tests.test_cluster.test_base
# Test the cluster base visualizers.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  Thu Mar 23 17:38:42 2017 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_base.py [241edca] benjamin@bengfort.com $

"""
Test the cluster base visualizers.
"""

##########################################################################
## Imports
##########################################################################

import unittest

from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.cluster.base import ClusteringScoreVisualizer

from sklearn.svm import SVC, SVR
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation
from sklearn.cluster import MeanShift, DBSCAN, Birch

##########################################################################
## Clustering Base Test Cases
##########################################################################

class ClusterBaseTests(unittest.TestCase):

    def test_clusterer_enforcement(self):
        """
        Assert that only clustering estimators can be passed to cluster viz
        """
        nomodels = [
            SVC, SVR, Ridge, RidgeCV, LinearRegression, RandomForestClassifier
        ]

        for nomodel in nomodels:
            with self.assertRaises(YellowbrickTypeError):
                ClusteringScoreVisualizer(nomodel())

        models = [
            KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, DBSCAN, Birch
        ]

        for model in models:
            try:
                ClusteringScoreVisualizer(model())
            except YellowbrickTypeError:
                self.fail("could not pass clustering estimator to visualizer")
