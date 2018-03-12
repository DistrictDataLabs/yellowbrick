# tests.test_features.test_rankd
# Test the rankd feature analysis visualizers
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Oct 07 12:19:19 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_rankd.py [01d5996] benjamin@bengfort.com $

"""
Test the Rankd feature analysis visualizers
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin
from yellowbrick.features.rankd import *

##########################################################################
## Rank1D Base Tests
##########################################################################


class Rank1DTests(VisualTestCase, DatasetMixin):
    X = np.array(
            [[ 2.318, 2.727, 4.260, 7.212, 4.792],
             [ 2.315, 2.726, 4.295, 7.140, 4.783,],
             [ 2.315, 2.724, 4.260, 7.135, 4.779,],
             [ 2.110, 3.609, 4.330, 7.985, 5.595,],
             [ 2.110, 3.626, 4.330, 8.203, 5.621,],
             [ 2.110, 3.620, 4.470, 8.210, 5.612,]]
        )

    y = np.array([1, 1, 0, 1, 0, 0])

    def setUp(self):
        super(Rank1DTests, self).setUp()
        self.occupancy = self.load_data('occupancy')

    def tearDown(self):
        super(Rank1DTests, self).tearDown()
        self.occupancy = None


    def test_rankd1(self):
        """
        Assert no errors occur during rand1 visualizer integration
        """
        visualizer = Rank1D()
        visualizer.fit_transform(self.X, self.y)
        visualizer.poof()
        self.assert_images_similar(visualizer)

    def test_integrated_rankd1(self):
        """
        Test rand1 on the real, occupancy data set
        """

        # Load the data from the fixture
        X = self.occupancy[[
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]]
        X = X.copy().view((float, len(X.dtype.names)))
        y = self.occupancy['occupancy'].astype(int)

        # Test the visualizer
        visualizer = Rank1D()
        visualizer.fit_transform(X, y)
        visualizer.poof()
        self.assert_images_similar(visualizer)


##########################################################################
## Rank2D Base Tests
##########################################################################

class Rank2DTests(VisualTestCase, DatasetMixin):
    X = np.array(
            [[ 2.318, 2.727, 4.260, 7.212, 4.792],
             [ 2.315, 2.726, 4.295, 7.140, 4.783,],
             [ 2.315, 2.724, 4.260, 7.135, 4.779,],
             [ 2.110, 3.609, 4.330, 7.985, 5.595,],
             [ 2.110, 3.626, 4.330, 8.203, 5.621,],
             [ 2.110, 3.620, 4.470, 8.210, 5.612,]]
        )

    y = np.array([1, 1, 0, 1, 0, 0])

    def setUp(self):
        super(Rank2DTests, self).setUp()
        self.occupancy = self.load_data('occupancy')

    def tearDown(self):
        super(Rank2DTests, self).tearDown()
        self.occupancy = None

    def test_rankd2(self):
        """
        Assert no errors occur during rand2 visualizer integration
        """
        visualizer = Rank2D()
        visualizer.fit_transform(self.X, self.y)
        visualizer.poof()


    @pytest.mark.xfail
    def test_integrated_rankd2(self):
        """
        Test rand2 on the real, occupancy data set
        """

        # Load the data from the fixture
        X = self.occupancy[[
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]]
        X = X.copy().view((float, len(X.dtype.names)))
        y = self.occupancy['occupancy'].astype(int)

        # Test the visualizer
        visualizer = Rank2D()
        visualizer.fit_transform(X, y)
        visualizer.poof()
        self.assert_images_similar(visualizer)
#
