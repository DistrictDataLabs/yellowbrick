# tests.test_text.test_correlation
# Tests for the dispersion plot visualization
#
# Author:   Patrick Deziel
# Created:  ----
#
# Copyright (C) 2022 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_correlation.py [25f1b9a] 

"""
Tests for the word correlation plot text visualization
"""

##########################################################################
## Imports
##########################################################################

from tests.base import VisualTestCase
from yellowbrick.datasets import load_hobbies
from yellowbrick.text.correlation import WordCorrelationPlot

##########################################################################
## Data
##########################################################################

corpus = load_hobbies()

##########################################################################
## WordCorrelationPlot Tests
##########################################################################

class TestWordCorrelationPlot(VisualTestCase):
    def test_word_correlation_plot(self):
        """
        Test WordCorrelationPlot on a text dataset
        """
        words = ['Game', 'player', 'score', 'oil', 'Man', 'woman']

        viz = WordCorrelationPlot(words)
        assert viz.fit(corpus.data) is viz, "fit method should return self"

        self.assert_images_similar(viz, tol=25)