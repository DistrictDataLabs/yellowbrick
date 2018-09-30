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
import pytest
from tests.base import VisualTestCase
from yellowbrick.contrib.missing.bar import *

try:
    import pandas as pd
except ImportError:
    pd = None

##########################################################################
## Feature Importances Tests
##########################################################################


@pytest.mark.usefixtures("missingdata")
class MissingBarVisualizerTestCase(VisualTestCase):
    """
    MissingBar Visualizer
    """

    def setUp(self):
        super(MissingBarVisualizerTestCase, self).setUp()
        self.tol = 0.01
        if os.name == 'nt': # Windows
            self.tol = 0.5

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_missingvaluesbar_pandas(self):
        """
        Integration test of visualizer with pandas
        """

        X_ = pd.DataFrame(self.missingdata.X)

        features = [str(n) for n in range(20)]
        viz = MissingValuesBar(features=features)
        viz.fit(X_)
        viz.poof()

        self.assert_images_similar(viz, tol=self.tol)

    def test_missingvaluesbar_numpy(self):
        """
        Integration test of visualizer with numpy without target y passed in
        """

        features = [str(n) for n in range(20)]
        viz = MissingValuesBar(features=features)
        viz.fit(self.missingdata.X)
        viz.poof()

        self.assert_images_similar(viz, tol=self.tol)


    def test_missingvaluesbar_numpy_with_y_target(self):
        """
        Integration test of visualizer with numpy without target y passed in
        but no class labels
        """

        features = [str(n) for n in range(20)]
        viz = MissingValuesBar(features=features)
        viz.fit(self.missingdata.X, self.missingdata.y)
        viz.poof()

        self.assert_images_similar(viz, tol=self.tol)


    def test_missingvaluesbar_numpy_with_y_target_with_labels(self):
        """
        Integration test of visualizer with numpy without target y passed in
        but no class labels
        """

        # add nan values to a range of values in the matrix

        features = [str(n) for n in range(20)]
        viz = MissingValuesBar(features=features, classes=['class A', 'class B'])
        viz.fit(self.missingdata.X, self.missingdata.y)
        viz.poof()

        self.assert_images_similar(viz, tol=self.tol)
