# -*- coding: utf-8 -*-
# tests.test_features.test_decomposition
# Tests for the ExplainedVariance feature visualizer.
#
# Author:   [Your Name]
# Created:  [Today's Date]
#
# Copyright (C) 2024 The scikit-yb developers
# For license information, see LICENSE.txt

"""
Tests for the ExplainedVariance visualizer.
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np

from tests.base import VisualTestCase
from yellowbrick.features.decomposition import ExplainedVariance
from yellowbrick.features.decomposition import explained_variance_visualizer


##########################################################################
## ExplainedVariance Tests
##########################################################################


class TestExplainedVariance(VisualTestCase):
    """
    Tests for the ExplainedVariance visualizer.
    """

    @pytest.fixture(autouse=True)
    def make_data(self):
        """Create simple dataset for all tests."""
        np.random.seed(42)
        self.X = np.random.rand(100, 5)

    def test_fit_and_transform(self):
        """
        Test that fit and transform work without errors.
        """
        viz = ExplainedVariance()
        viz.fit(self.X)
        transformed = viz.transform(self.X)
        assert transformed.shape[0] == self.X.shape[0]

    def test_explained_variance_property(self):
        """
        Test that explained_variance_ is accessible after fit.
        """
        viz = ExplainedVariance()
        viz.fit(self.X)
        ev = viz.explained_variance_
        assert ev is not None
        assert len(ev) == self.X.shape[1]
        # All variances should be positive
        assert all(v > 0 for v in ev)

    def test_n_components(self):
        """
        Test that n_components limits output dimensions.
        """
        n = 3
        viz = ExplainedVariance(n_components=n)
        viz.fit(self.X)
        transformed = viz.transform(self.X)
        assert transformed.shape == (self.X.shape[0], n)
        assert len(viz.explained_variance_) == n

    def test_scale_false(self):
        """
        Test that disabling scale works without error.
        """
        viz = ExplainedVariance(scale=False)
        viz.fit(self.X)
        viz.transform(self.X)

    def test_center_false(self):
        """
        Test that disabling center works without error.
        """
        viz = ExplainedVariance(center=False)
        viz.fit(self.X)
        viz.transform(self.X)

    def test_scale_and_center_false(self):
        """
        Test with both scale and center disabled.
        """
        viz = ExplainedVariance(scale=False, center=False)
        viz.fit(self.X)
        viz.transform(self.X)

    def test_quick_method(self):
        """
        Test the explained_variance_visualizer quick method.
        """
        viz = explained_variance_visualizer(self.X)
        assert viz is not None