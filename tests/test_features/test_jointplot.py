# tests.test_features.test_jointplot
# Test the JointPlotVisualizer
#
# Author:   Prema Damodaran Roman
# Created:  
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_jointplot.py

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

	
	def setUp(self):
		self.concrete = self.load_data('concrete')

	def tearDown(self):
		self.concrete = None

	def test_jointplot(self):
		"""
		Test that if X has more than 1 column 
		a value error would be raised
		"""
		X = np.array([1, 2, 3, 5, 8, 10, 2],
			  		 [2, 3, 5, 6, 7, 8, 9])
		y = np.array([1, 3, 6, 2, 9, 2, 3])
		
		with self.assertRaises(YellowbrickValueError):
			visualizer = JointPlotVisualizer()
			visualizer.fit(self.X, self.y)

	def test_jointplot(self):
		"""
		Test jointplot on the concrete data set
		"""

		# Load the data from the fixture
		X = self.concrete['cement'].as_matrix()
		y = self.concrete['strength'].as_matrix()

		# Test the visualizer
		visualizer = JointPlotVisualizer(feature=feature, target=target, joint_plot="hex")
		visualizer.fit(X, y)                # Fit the data to the visualizer
		g = visualizer.poof()