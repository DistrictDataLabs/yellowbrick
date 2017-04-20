# tests.test_features.test_jointplot
# Test the JointPlotVisualizer
#
# Author:   Prema Damodaran Roman
# Created:  Mon Apr 10 21:00:54 2017 -0400
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_jointplot.py [] pdamo24@gmail.com $

"""
Test the JointPlotVisualizer.

These tests work differently depending on what version of matplotlib is
installed. If version 2.0.0 or greater is installed, then most tests will
execute, otherwise most will skip and only the warning will be tested.
"""

##########################################################################
## Imports
##########################################################################

import warnings
import unittest
import numpy as np
import matplotlib as mpl
import numpy.testing as npt

from tests.dataset import DatasetMixin
from yellowbrick.features.jointplot import *

##########################################################################
## JointPlotVisualizer Tests
##########################################################################

# Determine version of matplotlib
MPL_VERS_MAJ = int(mpl.__version__.split(".")[0])


class JointPlotTests(unittest.TestCase, DatasetMixin):

    X = np.array([1, 2, 3, 5, 8, 10])

    y = np.array([1, 3, 6, 2, 9, 2])

    def setUp(self):
        self.concrete = self.load_data('concrete')

    def tearDown(self):
        self.concrete = None

    @unittest.skipIf(MPL_VERS_MAJ > 1, "requires matplotlib 1.5.3 or less")
    def test_warning(self):
        """
        Ensure that the jointplot warns if mpl version is < 2.0.0
        """
        # Note Python 3.2+ has a self.assertWarns ... but we need to be
        # Python 2.7 compatible, so we're going to do this. 
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            # Trigger a warning.
            visualizer = JointPlotVisualizer()

            # Ensure that a warning occurred
            self.assertEqual(len(w), 1)
            self.assertEqual(
                str(w[-1].message),
                "JointPlotVisualizer requires Matplotlib version 2.0.0.Please upgrade to continue."
            )


    @unittest.skipIf(MPL_VERS_MAJ < 2, "requires matplotlib 2.0.0 or greater")
    def test_jointplot(self):
        """
        Assert no errors occur during jointplot visualizer integration
        """

        visualizer = JointPlotVisualizer()
        visualizer.fit(self.X, self.y)
        visualizer.poof()


    @unittest.skipIf(MPL_VERS_MAJ < 2, "requires matplotlib 2.0.0 or greater")
    def test_jointplot_integrated(self):
        """
        Test jointplot on the concrete data set
        """

        # Load the data from the fixture
        X = self.concrete['cement']
        y = self.concrete['strength']
        feature = 'cement'
        target = 'strength'

        # Test the visualizer
        visualizer = JointPlotVisualizer(feature=feature, target=target, joint_plot="hex")
        visualizer.fit(X, y)                # Fit the data to the visualizer
        g = visualizer.poof()
