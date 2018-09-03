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

import pytest

from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.text.dispersion import *
from tests.dataset import DatasetMixin
from tests.base import VisualTestCase

##########################################################################
## DispersionPlot Tests
##########################################################################

class DispersionPlotTests(VisualTestCase, DatasetMixin):

    def test_integrated_dispersionplot(self):
        """
        Assert no errors occur during DispersionPlot integration
        """
        corpus = self.load_data('hobbies')

        text = [doc.split() for doc in corpus.data]
        target_words = ['Game', 'player', 'score', 'oil', 'Man']

        visualizer = DispersionPlot(target_words)
        visualizer.fit(text)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer, tol=25)

    def test_dispersionplot_ignore_case(self):
        """
        Assert no errors occur during DispersionPlot integration
        with ignore_case parameter turned on
        """
        corpus = self.load_data('hobbies')

        text = [doc.split() for doc in corpus.data]
        target_words = ['Game', 'player', 'score', 'oil', 'Man']

        visualizer = DispersionPlot(target_words, ignore_case=True)
        visualizer.fit(text)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer, tol=25)

    def test_dispersionplot_generator_input(self):
        """
        Assert no errors occur during dispersionPlot integration
        when the corpus' text type is a generator
        """
        corpus = self.load_data('hobbies')

        text = (doc.split() for doc in corpus.data)
        target_words = ['Game', 'player', 'score', 'oil', 'Man']

        visualizer = DispersionPlot(target_words, ignore_case=True)
        visualizer.fit(text)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer, tol=25)

    def test_dispersionplot_annotate_docs(self):
        """
        Assert no errors occur during DispersionPlot integration
        with annotate_docs parameter turned on
        """
        corpus = self.load_data('hobbies')

        text = [doc.split() for doc in corpus.data]
        target_words = ['girl', 'she', 'boy', 'he', 'man']

        visualizer = DispersionPlot(target_words, annotate_docs=True)
        visualizer.fit(text)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer, tol=25)

    def test_dispersionplot_color_words_by_class(self):
        """
        Assert no errors occur during DispersionPlot integration
        when target values are specified
        """
        corpus = self.load_data('hobbies')

        text = (doc.split() for doc in corpus.data)
        target_words = ['girl', 'she', 'boy', 'he', 'man']

        target_values = corpus.target

        visualizer = DispersionPlot(target_words)
        visualizer.fit(text, target_values)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer, tol=25)

    def test_dispersionplot_mismatched_labels(self):
        """
        Assert exception is raised when number of labels doesn't match
        """
        corpus = self.load_data('hobbies')

        text = (doc.split() for doc in corpus.data)
        target_words = ['girl', 'she', 'boy', 'he', 'man']

        target_values = corpus.target

        visualizer = DispersionPlot(target_words, annotate_docs=True,
                                    labels=['a', 'b'])

        msg = (
            r'number of supplied labels \(\d\) '
            r'does not match the number of classes \(\d\)'
        )

        with pytest.raises(YellowbrickValueError, match=msg):
            visualizer.fit(text, target_values)
