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
import os
import pytest
from tests.base import VisualTestCase
from yellowbrick.contrib.missing.dispersion import *

try:
    import pandas as pd
except ImportError:
    pd = None

##########################################################################
## Feature Importances Tests
##########################################################################

@pytest.mark.usefixtures("missingdata")
class MissingValuesDispersionTestCase(VisualTestCase):
    """
    MissingValuesDispersion visualizer
    """
    def setUp(self):
        super(MissingValuesDispersionTestCase, self).setUp()
        self.tol = 0.01
        if os.name == 'nt': # Windows
            self.tol = 5.0

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_viz_properties(self):
        """
        Integration test of visualizer with pandas
        """

        X_ = pd.DataFrame(self.missingdata.X)
        features = [str(n) for n in range(20)]
        viz = MissingValuesDispersion(features=features)

        assert viz.nan_locs == []


    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_missingvaluesdispersion_with_pandas(self):
        """
        Integration test of visualizer with pandas
        """

        X_ = pd.DataFrame(self.missingdata.X)
        features = [str(n) for n in range(20)]
        viz = MissingValuesDispersion(features=features)
        viz.fit(X_)
        viz.poof()

        self.assert_images_similar(viz, tol=self.tol)

    def test_missingvaluesdispersion_with_pandas_with_y_targets(self):
        """
        Integration test of visualizer with pandas with y targets
        """

        X_ = pd.DataFrame(self.missingdata.X)
        features = [str(n) for n in range(20)]
        classes = ['Class A', 'Class B']
        viz = MissingValuesDispersion(features=features, classes=classes)
        viz.fit(X_, y=self.missingdata.y)
        viz.poof()

        self.assert_images_similar(viz, tol=self.tol)


    def test_missingvaluesdispersion_with_numpy(self):
        """
        Integration test of visualizer with numpy
        """
        features = [str(n) for n in range(20)]
        viz = MissingValuesDispersion(features=features)
        viz.fit(self.missingdata.X)
        viz.poof()

        self.assert_images_similar(viz, tol=self.tol)

    def test_missingvaluesdispersion_with_numpy_with_y_targets(self):
        """
        Integration test of visualizer with numpy with y targets
        """

        features = [str(n) for n in range(20)]
        classes = ['Class A', 'Class B']
        viz = MissingValuesDispersion(features=features, classes=classes)
        viz.fit(self.missingdata.X, y=self.missingdata.y)
        viz.poof()

        self.assert_images_similar(viz, tol=self.tol)
