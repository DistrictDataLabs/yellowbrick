# tests.test_missing.test_bar
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
import unittest
import numpy.testing as npt

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin
from yellowbrick.missing.bar import *

try:
    import pandas
except ImportError:
    pandas = None

##########################################################################
## Feature Importances Tests
##########################################################################

class TestFeatureImportancesVisualizer(VisualTestCase, DatasetMixin):
    """
    FeatureImportances visualizer
    """

    def test_integration_feature_importances(self):
        """
        Integration test of visualizer with feature importances param
        """
        mushrooms = self.load_data('mushroom')
        features = ['shape', 'surface', 'color']
        target   = ['target']
        X = mushrooms[features].as_matrix()
        y = mushrooms[target].as_matrix()

        viz = MissingValuesBar(features=features)
        viz.fit(X, y=y)
        viz.poof()

        self.assert_images_similar(viz)
