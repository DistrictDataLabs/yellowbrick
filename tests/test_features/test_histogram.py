import pytest
from tests.base import VisualTestCase
from tests.dataset import DatasetMixin
from yellowbrick.features.histogram import *

##########################################################################
## BalancedBinningReference Tests
##########################################################################

class BalancedBinningReferenceTests(VisualTestCase, DatasetMixin):

    def test_balancedbinningreference(self):
        """
        Test Histogram on a real dataset
        """
        # Load the data from the fixture
        dataset = self.load_data('occupancy')

        # Get the data
        X = dataset["temperature"]

        try:
            visualizer = BalancedBinningReference()
            visualizer.fit(X)
            visualizer.poof()
        except Exception as e:
            pytest.fail("my visualizer didn't work")
