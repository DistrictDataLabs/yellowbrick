# tests.test_text.test_freqdist
# Tests for the frequency distribution visualization
#
# Author:   Rebecca Bilbro
# Github:   @rebeccabilbro
# Created:  2017-03-22 15:27
#
# Copyright (C) 2018
# For license information, see LICENSE.txt
#
# ID: test_freqdist.py [bd9cbb9] rbilbro@districtdatalabs.com $

"""
Tests for the frequency distribution text visualization
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest

from yellowbrick.text.dispersion import *
from tests.dataset import DatasetMixin
from tests.base import VisualTestCase


##########################################################################
## DispersionPlot Tests
##########################################################################

class FreqDistTests(VisualTestCase, DatasetMixin):

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_integrated_dispersionplot(self):
        """
        Assert no errors occur during DispersionPlot integration
        """
        corpus = self.load_data('hobbies')
	
        docs = ' '.join(corpus.data)
        text = docs.split()

        target_words = ['game', 'player', 'score', 'oil', 'man']

        visualizer = DispersionPlot(target_words)
        visualizer.fit(text)

        visualizer.poof()
        self.assert_images_similar(visualizer)
