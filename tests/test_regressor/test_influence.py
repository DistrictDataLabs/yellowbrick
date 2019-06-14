# tests.test_regressor.test_influence
# Test the regressor influence visualizers.
#
# Author:   Benjamin Bengfort <benjamin@bengfort.com>
# Created:  Sun Jun 09 16:03:31 2019 -0400
#
# Copyright (C) 2019 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_influence.py [] benjamin@bengfort.com $

"""
Test the regressor influence visualizers.
"""

##########################################################################
## Imports
##########################################################################

import matplotlib.pyplot as plt

from tests.base import VisualTestCase
from sklearn.datasets import make_regression

from yellowbrick.regressor.influence import *


##########################################################################
## Test CooksDistance Visualizer
##########################################################################

class TestCooksDistance(VisualTestCase):

    def test_cooks_distance(self):
        """
        Test image similarity of Cook's Distance on a random dataset
        """
        # Make Test Dataset
        X, y = make_regression(
            n_samples=100, n_features=14, n_informative=6, bias=1.2,
            noise=49.8, tail_strength=0.6, random_state=637
        )

        _, ax = plt.subplots()
        viz = CooksDistance(ax=ax)

        assert viz.fit(X, y) is viz
        self.assert_images_similar(viz)

    def test_cooks_distance_quickmethod(self):
        """
        Test the cooks_distance quick method on a random dataset
        """
        # Make Test Dataset
        X, y = make_regression(
            n_samples=100, n_features=14, n_informative=6, bias=1.2,
            noise=49.8, tail_strength=0.6, random_state=637
        )

        _, ax = plt.subplots()
        viz = cooks_distance(X, y, ax=ax, influence_threshold=False)
        self.assert_images_similar(viz)