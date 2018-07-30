# tests.test_target.test_binning
# Tests for the BalancedBinningReference visualizer
#
# Author:   Juan L. Kehoe (juanluo2008@gmail.com)
# Author:  Prema Damodaran Roman (pdamo24@gmail.com)
# Created: Thu Jul 20 10:21:49 2018 -0400
#
# ID: test_binning.py

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin
from yellowbrick.target.binning import *

##########################################################################
## BalancedBinningReference Tests
##########################################################################

class TestBalancedBinningReference(VisualTestCase, DatasetMixin):
	"""
	Test the BalancedBinningReference visualizer 
	"""

	def test_balancedbinningreference(self):
		"""
		Test Histogram on a real dataset
		"""
		# Load the data from the fixture
		dataset = self.load_data('occupancy')

		# Get the data
		y = dataset["temperature"]

		
		visualizer = BalancedBinningReference()
		visualizer.fit(y)
		visualizer.poof()
		self.assert_images_similar(visualizer, tol=0.5)
			
		