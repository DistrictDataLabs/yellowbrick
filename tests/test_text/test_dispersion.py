# tests.test_text.test_dispersion
# Tests for the dispersion plot visualization
#
# Author:   Larry Gray
# Github:   @lwgray
# Created:  2018-06-22 15:27
#
# Copyright (C) 2018
# For license information, see LICENSE.txt
#
# ID: test_dispersion.py [] lwgray@gmail.com $

"""
Tests for the dispersion plot text visualization
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest

from yellowbrick.text.dispersion import *
from tests.dataset import DatasetMixin
from tests.base import VisualTestCase
from itertools import chain

##########################################################################
## DispersionPlot Tests
##########################################################################

@pytest.mark.xfail(sys.platform == "win32", reason="Issue #491")
class DispersionPlotTests(VisualTestCase, DatasetMixin):

    def test_integrated_dispersionplot(self):
        """
        Assert no errors occur during DispersionPlot integration
        """
        corpus = self.load_data('hobbies')
	
        text = [word for doc in corpus.data for word in doc.split()]
        target_words = ['Game', 'player', 'score', 'oil', 'Man']

        visualizer = DispersionPlot(target_words)
        visualizer.fit(text)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer, tol=12)

    def test_dispersionplot_ignore_case(self):
        """
        Assert no errors occur during DispersionPlot integration
        with ignore_case parameter turned on
        """
        corpus = self.load_data('hobbies')
	
        text = [word for doc in corpus.data for word in doc.split()]
        target_words = ['Game', 'player', 'score', 'oil', 'Man']

        visualizer = DispersionPlot(target_words, ignore_case=True)
        visualizer.fit(text)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer, tol=12)

    def test_dispersionplot_generator_input(self):
        """
        Assert no errors occur during dispersionPlot integration
        when the corpus' text type is a generator
        """
        corpus = self.load_data('hobbies')

        text = chain(*map(str.split, corpus.data))
        target_words = ['Game', 'player', 'score', 'oil', 'Man']

        visualizer = DispersionPlot(target_words, ignore_case=True)
        visualizer.fit(text)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer, tol=12)
        
