# tests.test_target.test_binning
# Tests for the BalancedBinningReference visualizer
#
# Author:  Juan L. Kehoe
# Author:  Prema Damodaran Roman
# Created: Thu Jul 20 10:21:49 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_binning.py

##########################################################################
# Imports
##########################################################################
import pytest

from yellowbrick.target.binning import *
from yellowbrick.datasets import load_occupancy

from tests.base import VisualTestCase

try:
    import pandas as pd
except ImportError:
    pd = None

##########################################################################
# BalancedBinningReference Tests
##########################################################################


class TestBalancedBinningReference(VisualTestCase):
    """
    Test the BalancedBinningReference visualizer
    """

    def test_numpy_bins(self):
        """
        Test Histogram on a NumPy array
        """
        # Load the data from the fixture
        data = load_occupancy(return_dataset=True)
        X, y = data.to_numpy()

        visualizer = BalancedBinningReference()
        visualizer.fit(y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.5)

    @pytest.mark.skipif(pd is None, reason="pandas is required")
    def test_pandas_bins(self):
        """
        Test Histogram on a Pandas Dataframe
        """
        # Load the data from the fixture
        data = load_occupancy(return_dataset=True)
        X, y = data.to_pandas()

        visualizer = BalancedBinningReference()
        visualizer.fit(y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.5)

    def test_quick_method(self):
        """
        Test the quick method with producing a valid visualization
        """
        data = load_occupancy(return_dataset=True)
        _, y = data.to_numpy()

        visualizer = balanced_binning_reference(y, show=False)

        assert isinstance(visualizer, BalancedBinningReference)
        self.assert_images_similar(visualizer, tol=0.5)
