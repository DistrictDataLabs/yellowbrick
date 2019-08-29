# tests.test_text.test_dispersion
# Tests for the dispersion plot visualization
#
# Author:   Larry Gray
# Created:  2018-06-22 15:27
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_dispersion.py [25f1b9a] lwgray@gmail.com $

"""
Tests for the dispersion plot text visualization
"""

##########################################################################
## Imports
##########################################################################

import pytest
import matplotlib.pyplot as plt

from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.datasets import load_hobbies
from yellowbrick.text.dispersion import *
from tests.base import VisualTestCase


##########################################################################
## Data
##########################################################################

corpus = load_hobbies()

##########################################################################
## DispersionPlot Tests
##########################################################################


class TestDispersionPlot(VisualTestCase):
    def test_quick_method(self):
        """
        Assert no errors occur when using the qucik method
        """
        _, ax = plt.subplots()

        text = [doc.split() for doc in corpus.data]
        target_words = ["Game", "player", "score", "oil", "Man"]

        viz = dispersion(words=target_words, corpus=text, ax=ax)
        viz.ax.grid(False)

        self.assert_images_similar(viz, tol=25)

    def test_integrated_dispersion_plot(self):
        """
        Assert no errors occur during DispersionPlot integration
        """
        text = [doc.split() for doc in corpus.data]
        target_words = ["Game", "player", "score", "oil", "Man"]

        visualizer = DispersionPlot(target_words)
        visualizer.fit(text)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer, tol=25)

    def test_dispersion_plot_ignore_case(self):
        """
        Assert no errors occur during DispersionPlot integration
        with ignore_case parameter turned on
        """
        text = [doc.split() for doc in corpus.data]
        target_words = ["Game", "player", "score", "oil", "Man"]

        visualizer = DispersionPlot(target_words, ignore_case=True)
        visualizer.fit(text)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer, tol=25)

    def test_dispersion_plot_generator_input(self):
        """
        Assert no errors occur during dispersionPlot integration
        when the corpus' text type is a generator
        """
        text = [doc.split() for doc in corpus.data]
        target_words = ["Game", "player", "score", "oil", "Man"]

        visualizer = DispersionPlot(target_words, ignore_case=True)
        visualizer.fit(text)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer, tol=25)

    def test_dispersion_plot_annotate_docs(self):
        """
        Assert no errors occur during DispersionPlot integration
        with annotate_docs parameter turned on
        """
        text = [doc.split() for doc in corpus.data]
        target_words = ["girl", "she", "boy", "he", "man"]

        visualizer = DispersionPlot(target_words, annotate_docs=True)
        visualizer.fit(text)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer, tol=25.5)

    def test_dispersion_plot_color_by_class(self):
        """
        Assert no errors occur during DispersionPlot integration
        when target values are specified
        """
        target = corpus.target
        text = [doc.split() for doc in corpus.data]
        target_words = ["girl", "she", "boy", "he", "man"]

        visualizer = DispersionPlot(target_words)
        visualizer.fit(text, target)
        visualizer.ax.grid(False)

        self.assert_images_similar(visualizer, tol=25)

    def test_dispersion_plot_mismatched_labels(self):
        """
        Assert exception is raised when number of labels doesn't match
        """
        target = corpus.target
        text = [doc.split() for doc in corpus.data]
        target_words = ["girl", "she", "boy", "he", "man"]

        visualizer = DispersionPlot(target_words, annotate_docs=True, labels=["a", "b"])

        msg = (
            r"number of supplied labels \(\d\) "
            r"does not match the number of classes \(\d\)"
        )

        with pytest.raises(YellowbrickValueError, match=msg):
            visualizer.fit(text, target)
