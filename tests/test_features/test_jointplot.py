# tests.test_features.test_jointplot
# Test the JointPlotVisualizer
#
# Author:   Prema Damodaran Roman
# Created:  Mon Apr 10 21:00:54 2017 -0400
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_jointplot.py [9e008b0] pdamodaran@users.noreply.github.com $

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
import matplotlib.pyplot as plt

from tests.dataset import DatasetMixin
from tests.base import VisualTestCase
from yellowbrick.features.jointplot import *

##########################################################################
## JointPlotVisualizer Tests
##########################################################################

# Determine version of matplotlib
MPL_VERS_MAJ = int(mpl.__version__.split(".")[0])


class JointPlotTests(VisualTestCase, DatasetMixin):

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
            JointPlotVisualizer()

            # Ensure that a warning occurred
            self.assertEqual(len(w), 1)
            self.assertEqual(
                str(w[-1].message),
                "JointPlotVisualizer requires matplotlib major version 2 "
                "or greater. Please upgrade."
            )


    @unittest.skipIf(MPL_VERS_MAJ < 2, "requires matplotlib 2.0.0 or greater")
    def test_jointplot_has_no_errors(self):
        """
        Assert no errors occur during jointplot visualizer integration
        """
        fig = plt.figure()
        ax = fig.add_subplot()

        visualizer = JointPlotVisualizer(ax=ax)
        visualizer.fit(self.X, self.y)
        visualizer.poof()

        self.assert_images_similar(visualizer)


    @unittest.skipIf(MPL_VERS_MAJ < 2, "requires matplotlib 2.0.0 or greater")
    def test_jointplot_integrated_has_no_errors(self):
        """
        Test jointplot on the concrete data set
        """

        fig = plt.figure()
        ax = fig.add_subplot()

        # Load the data from the fixture
        X = self.concrete['cement']
        y = self.concrete['strength']
        feature = 'cement'
        target = 'strength'

        # Test the visualizer
        visualizer = JointPlotVisualizer(
            feature=feature, target=target, joint_plot="hex", ax=ax)
        visualizer.fit(X, y)
        visualizer.poof()

        self.assert_images_similar(visualizer)


    @unittest.skipIf(MPL_VERS_MAJ < 2, "requires matplotlib 2.0.0 or greater")
    def test_jointplot_no_matplotlib2_warning(self):
        """
        Assert no UserWarning occurs if matplotlib major version >= 2
        """
        with warnings.catch_warnings(record=True) as ws:
            # Filter on UserWarnings
            warnings.filterwarnings("always", category=UserWarning)
            visualizer = JointPlotVisualizer()
            visualizer.fit(self.X, self.y)
            visualizer.poof()

            # Filter out user warnings not related to matplotlib version
            ver_warn_msg = "requires matplotlib major version 2 or greater"
            mpl_ver_cnt = 0
            for w in ws:
                if w and w.message and ver_warn_msg in str(w.message):
                    mpl_ver_cnt += 1
            self.assertEqual(0, mpl_ver_cnt, ws[-1].message \
                        if ws else "No error")
