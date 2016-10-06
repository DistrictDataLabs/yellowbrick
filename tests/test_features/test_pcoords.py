# tests.test_features.test_pcoords
# Testing for the parallel coordinates feature visualizers
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Oct 06 11:21:27 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_pcoords.py [] benjamin@bengfort.com $

"""
Testing for the parallel coordinates feature visualizers
"""

##########################################################################
## Imports
##########################################################################

import unittest
import numpy as np

from yellowbrick.features.pcoords import *


##########################################################################
## Parallel Coordinates Tests
##########################################################################


class ParallelCoordinatesTests(unittest.TestCase):

    X = np.array(
            [[ 2.318, 2.727, 4.260, 7.212, 4.792],
             [ 2.315, 2.726, 4.295, 7.140, 4.783,],
             [ 2.315, 2.724, 4.260, 7.135, 4.779,],
             [ 2.110, 3.609, 4.330, 7.985, 5.595,],
             [ 2.110, 3.626, 4.330, 8.203, 5.621,],
             [ 2.110, 3.620, 4.470, 8.210, 5.612,]]
        )

    y = np.array([1, 1, 0, 1, 0, 0])

    def test_parallel_coords(self):
        """
        Assert no errors occur during parallel coordinates integration
        """
        visualizer = ParallelCoordinates()
        visualizer.fit_transform(self.X, self.y)
