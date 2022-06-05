# tests.test_text.test_correlation
# Tests for the dispersion plot visualization
#
# Author:   Patrick Deziel
# Created:  Tue May 3 16:18:21 2022 -0500
#
# Copyright (C) 2022 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_correlation.py [0a7d2fe] deziel.patrick@gmail.com $

"""
Tests for the word correlation plot text visualization
"""

##########################################################################
## Imports
##########################################################################

import pytest
import matplotlib.pyplot as plt

from tests.base import VisualTestCase
from yellowbrick.datasets import load_hobbies
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.text.correlation import WordCorrelationPlot, word_correlation

##########################################################################
## Data
##########################################################################

corpus = load_hobbies()

##########################################################################
## WordCorrelationPlot Tests
##########################################################################

class TestWordCorrelationPlot(VisualTestCase):
    def test_quick_method(self):
        """
        Assert no errors occur when using the quick method.
        """
        _, ax = plt.subplots()
        words = ["Game", "player", "score", "oil"]

        viz = word_correlation(words, corpus.data, ax=ax, show=False)

        self.assert_images_similar(viz, tol=25)

    def test_word_correlation_plot(self):
        """
        Assert no errors are raised during normal execution.
        """
        words = ["Game", "player", "score", "oil", "Man"]

        viz = WordCorrelationPlot(words)
        assert viz.fit(corpus.data) is viz, "fit method should return self"

        self.assert_images_similar(viz, tol=25)

    def test_word_correlation_generator(self):
        """
        Assert no errors are raised when the corpus is a generator.
        """
        words = ["Game", "player", "score", "oil", "Man", "woman"]

        viz = WordCorrelationPlot(words)

        def stream_corpus(data):
            for doc in data:
                yield doc
        viz.fit(stream_corpus(corpus.data))

        self.assert_images_similar(viz, tol=25)

    def test_word_correlation_ignore_case(self):
        """
        Assert no errors are raised when ignore_case is True.
        """
        words = ["Game", "player", "score", "oil", "game"]

        viz = WordCorrelationPlot(words, ignore_case=True)
        viz.fit(corpus.data)

        self.assert_images_similar(viz, tol=25)

    def test_word_correlation_ngrams(self):
        """
        Assert no errors are raised when multiple-word terms are provided.
        """
        words = ["Tatsumi Kimishima", "Nintendo", "game", "man", "play"]

        viz = WordCorrelationPlot(words)
        viz.fit(corpus.data)

        self.assert_images_similar(viz, tol=25)

    def test_word_correlation_no_words(self):
        """
        Assert that an error is raised when no words are provided.
        """
        with pytest.raises(YellowbrickValueError):
            WordCorrelationPlot([])

        with pytest.raises(YellowbrickValueError):
            WordCorrelationPlot([""])

        with pytest.raises(YellowbrickValueError):
            WordCorrelationPlot([" ", "\t", "\n"])

    def test_word_correlation_missing_words(self):
        """
        Assert that an error is raised on fit() when a word does not exist in the
        corpus.
        """
        words = ["Game", "player", "score", "oil", "NotACorpusWord"]
        
        viz = WordCorrelationPlot(words)
        with pytest.raises(YellowbrickValueError):
            viz.fit(corpus.data)