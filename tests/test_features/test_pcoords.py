# tests.test_features.test_pcoords
# Testing for the parallel coordinates feature visualizers
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Oct 06 11:21:27 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_pcoords.py [1d407ab] benjamin@bengfort.com $

"""
Testing for the parallel coordinates feature visualizers
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from tests.base import VisualTestCase
from yellowbrick.features.pcoords import *
from tests.dataset import DatasetMixin

##########################################################################
## Parallel Coordinates Tests
##########################################################################


class ParallelCoordinatesTests(VisualTestCase, DatasetMixin):

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
        visualizer.poof()
        self.assert_images_similar(visualizer)


    def test_normalized_pcoords(self):
        """
        Assert no errors occur using 'normalize' argument
        """
        visualizer = ParallelCoordinates(normalize='l2')
        visualizer.fit_transform(self.X, self.y)
        visualizer.poof()
        self.assert_images_similar(visualizer)

    def test_normalized_pcoords_invalid_arg(self):
        """
        Invalid argument to 'normalize' should raise
        """
        with self.assertRaises(YellowbrickValueError):
            ParallelCoordinates(normalize='foo')

    def test_pcoords_sample_int(self):
        """
        Assert no errors occur using integer 'sample' argument
        """
        visualizer = ParallelCoordinates(sample=10)
        visualizer.fit_transform(self.X, self.y)

    def test_pcoords_sample_int_invalid(self):
        """
        Negative int values should raise
        """
        with self.assertRaises(YellowbrickValueError):
            ParallelCoordinates(sample=-1)

    def test_pcoords_sample_float(self):
        """
        Assert no errors occur using float 'sample' argument
        """
        visualizer = ParallelCoordinates(sample=0.5)
        visualizer.fit_transform(self.X, self.y)

    def test_pcoords_sample_float_invalid(self):
        """
        Float values for 'sample' argument outside [0,1] should raise.
        """
        with self.assertRaises(YellowbrickValueError):
            ParallelCoordinates(sample=-0.2)
        with self.assertRaises(YellowbrickValueError):
            ParallelCoordinates(sample=1.1)

    def test_pcoords_sample_invalid_type(self):
        """
        Non-numeric values for 'sample' argument should raise.
        """
        with self.assertRaises(YellowbrickTypeError):
            ParallelCoordinates(sample='foo')

    def test_integrated_pcoords(self):
        """
        Test parallel coordinates on a real data set (downsampled for speed)
        """
        occupancy = self.load_data('occupancy')

        X = occupancy[[
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]]

        y = occupancy['occupancy'].astype(int)

        # Convert X to an ndarray
        X = X.copy().view((float, len(X.dtype.names)))

        # Test the visualizer
        visualizer = ParallelCoordinates(sample=200)
        visualizer.fit_transform(X, y)
        visualizer.poof()
        self.assert_images_similar(visualizer)
