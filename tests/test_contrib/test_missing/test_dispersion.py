# tests.test_contrib.test_missing.test_dispersion
# Tests for the alpha selection visualizations.
#
# Author:  Nathan Danielsen <nathan.danielsen@gmail.com>
# Created:  Thu Mar 29 12:13:04 2018 -0500
#
# Copyright (C) 2018 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_dispersion.py [7d3f5e6] nathan.danielsen@gmail.com $

"""
Tests for the MissingValuesDispersion visualizations.
"""

##########################################################################
## Imports
##########################################################################
import pytest

from sklearn.datasets import make_classification

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin
from yellowbrick.contrib.missing.dispersion import *

try:
    import pandas as pd
except ImportError:
    pd = None

##########################################################################
## Feature Importances Tests
##########################################################################

class MissingValuesDispersionTestCase(VisualTestCase, DatasetMixin):
    """
    MissingValuesDispersion visualizer
    """

    def test_missingvaluesdispersion(self):
        """
        Integration test of visualizer with feature importances param
        """
        mushrooms = self.load_data('mushroom')
        features = ['shape', 'surface', 'color']
        target   = ['target']
        X = mushrooms[features].as_matrix()
        y = mushrooms[target].as_matrix()

        viz = MissingValuesDispersion(features=features)
        viz.fit(X, y=y)
        viz.poof()

        self.assert_images_similar(viz)


    def test_missingvaluesdispersion_with_pandas(self):
        """
        Integration test of visualizer with feature importances param
        """
        X, y = make_classification(
            n_samples=400, n_features=20, n_informative=8, n_redundant=8,
            n_classes=2, n_clusters_per_class=4, random_state=854
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        X_ = pd.DataFrame(X).as_matrix()
        features = [str(n) for n in range(20)]
        viz = MissingValuesDispersion(features=features)
        viz.fit(X_, y=y)
        viz.poof()

        self.assert_images_similar(viz)
