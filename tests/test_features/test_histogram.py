import pytest
from tests.base import VisualTestCase
from tests.dataset import DatasetMixin
from yellowbrick.features.histogram import *

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
		X = dataset["temperature"]

		
		visualizer = BalancedBinningReference()
		visualizer.fit(X)
		visualizer.poof()
		self.assert_images_similar(visualizer)
			
		