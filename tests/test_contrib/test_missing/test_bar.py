# tests.test_contrib.test_missing.test_bar
# Tests for the alpha selection visualizations.
#
# Author:  Nathan Danielsen <nathan.danielsen@gmail.com>
# Created:  Thu Mar 29 12:13:04 2018 -0500
#
# Copyright (C) 2018 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_bar.py [7d3f5e6] nathan.danielsen@gmail.com $

"""
Tests for the MissingValuesBar visualizations.
"""

##########################################################################
## Imports
##########################################################################

import os
from tests.base import VisualTestCase
from sklearn.datasets import make_classification
from yellowbrick.contrib.missing.bar import *

try:
    import pandas as pd
except ImportError:
    pd = None

##########################################################################
## Feature Importances Tests
##########################################################################

class TestMissingBarVisualizer(VisualTestCase):
    """
    FeatureImportances visualizer
    """

    def setUp(self):
        super(TestMissingBarVisualizer, self).setUp()
        self.tol = 0.01
        if os.name == 'nt': # Windows
            self.tol = 0.5

    def test_missingvaluesbar_pandas(self):
        """
        Integration test of visualizer with pandas
        """
        X, y = make_classification(
            n_samples=400, n_features=20, n_informative=8, n_redundant=8,
            n_classes=2, n_clusters_per_class=4, random_state=854
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan
        X_ = pd.DataFrame(X)

        features = [str(n) for n in range(20)]
        viz = MissingValuesBar(features=features)
        viz.fit(X_)
        viz.poof()

        self.assert_images_similar(viz, tol=self.tol)


    def test_missingvaluesbar_numpy(self):
        """
        Integration test of visualizer with numpy without target y passed in
        """
        X, y = make_classification(
            n_samples=400, n_features=20, n_informative=8, n_redundant=8,
            n_classes=2, n_clusters_per_class=4, random_state=856
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        features = [str(n) for n in range(20)]
        viz = MissingValuesBar(features=features)
        viz.fit(X)
        viz.poof()

        self.assert_images_similar(viz, tol=self.tol)

    def test_missingvaluesbar_numpy_with_y_target(self):
        """
        Integration test of visualizer with numpy without target y passed in
        but no class labels
        """
        X, y = make_classification(
            n_samples=400, n_features=20, n_informative=8, n_redundant=8,
            n_classes=2, n_clusters_per_class=4, random_state=856
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        features = [str(n) for n in range(20)]
        viz = MissingValuesBar(features=features)
        viz.fit(X, y)
        viz.poof()

        self.assert_images_similar(viz, tol=self.tol)

    def test_missingvaluesbar_numpy_with_y_target_with_labels(self):
        """
        Integration test of visualizer with numpy without target y passed in
        but no class labels
        """
        X, y = make_classification(
            n_samples=400, n_features=20, n_informative=8, n_redundant=8,
            n_classes=2, n_clusters_per_class=4, random_state=856
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        features = [str(n) for n in range(20)]
        viz = MissingValuesBar(features=features, classes=['class A', 'class B'])
        viz.fit(X, y)
        viz.poof()

        self.assert_images_similar(viz, tol=self.tol)
