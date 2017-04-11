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
Test the JointPlotVisualizer
"""

##########################################################################
## Imports
##########################################################################

import unittest
import numpy as np
import numpy.testing as npt

from tests.dataset import DatasetMixin
from yellowbrick.features.jointplot import *

##########################################################################
## JointPlotVisualizer Tests
##########################################################################

class JointPlotTests(unittest.TestCase, DatasetMixin):

	X = np.array([1, 2, 3, 5, 8, 10])

	y = np.array([1, 3, 6, 2, 9, 2])
	
	def setUp(self):
		self.concrete = self.load_data('concrete')

	def tearDown(self):
		self.concrete = None

	def test_jointplot(self):
		"""
		Assert no errors occur during jointplot visualizer integration
		"""
		
		visualizer = JointPlotVisualizer()
		visualizer.fit(self.X, self.y)
		visualizer.poof()
		
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
