# yellowbrick.text.correlation
# Implementation of word correlation for text visualization.
#
# Author:   Patrick Deziel
# Created:  ---
#
# Copyright (C) 2022 The scikit-yb developers
# For license information, see LICENSE.txt
#

"""
Implementation of word correlation for text visualization.
"""


##########################################################################
## Imports
##########################################################################

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from yellowbrick.style import find_text_color
from yellowbrick.exceptions import ModelError
from yellowbrick.text.base import TextVisualizer
from yellowbrick.style.palettes import color_sequence
from yellowbrick.exceptions import YellowbrickValueError

##########################################################################
## Word Correlation Plot Visualizer
##########################################################################

class WordCorrelationPlot(TextVisualizer):
    """
    Word correlation illustrates the extent to which words in a corpus appear in the
    same documents.

    WordCorrelationPlot visualizes the binary correlation between words across
    documents as a heatmap. The correlation is defined using the phi-coefficient or
    mean square contingency coefficient between any two words m and n.

    Parameters
    ----------


    Attributes
    ----------

    """
    def __init__(
        self,
        words,
        ignore_case=False,
        ngram_range=(1, 1),
        ax=None,
        cmap="RdYlBu",
        colorbar=True,
        fontsize=None,
        **kwargs
    ):
        super(WordCorrelationPlot, self).__init__(ax=ax, **kwargs)

        # Visual parameters
        self.fontsize = fontsize
        self.colorbar = colorbar
        self.cmap = color_sequence(cmap)

        # Fitting parameters
        self.ignore_case = ignore_case
        self.ngram_range = ngram_range
        self.words = self._construct_terms(words, ignore_case)

    def _construct_terms(self, words, ignore_case):
        """
        Constructs the list of terms to be plotted based on the provided words.
        """
        # Remove surrounding whitespace
        terms = [word.strip() for word in words if len(word.strip()) > 0]

        if len(terms) == 0:
            raise YellowbrickValueError("Must provide at least one word to plot.")

        # Convert to lowercase if ignore_case is set
        if ignore_case:
            terms = [word.lower() for word in terms]

        # Sort and remove duplicates
        return sorted(set(terms))

    def _compute_coefficient(self, m, n):
        """
        Computes the phi-coefficient for two words m and n.
        """
        m_col = self.doc_term_matrix_.getcol(self.term_index_[m])
        n_col = self.doc_term_matrix_.getcol(self.term_index_[n])
        both = m_col.multiply(n_col).sum()
        m_total = m_col.sum()
        n_total = n_col.sum()
        only_m = m_total - both
        only_n = n_total - both
        neither = self.docs_ - both - only_m - only_n
        return ((both * neither) - (only_m * only_n)) / np.sqrt(m_total * n_total * (self.docs_ - m_total) * (self.docs_ - n_total))

    def _construct_term_index(self):
        """
        Constructs a dictionary mapping terms to indices in the sparse doc-term matrix.
        """
        self.term_index_ = {}
        for i, term in enumerate(self.vocab_):
            if term in self.words:
                if term in self.term_index_:
                    raise ModelError("Duplicate term in vocabulary: {}".format(term))
                self.term_index_[term] = i
                if len(self.term_index_) == len(self.words):
                    break

        # Verify that all target words exist in the corpus
        for word in self.words:
            if self.doc_term_matrix_.getcol(self.term_index_[word]).sum() == 0:
                raise YellowbrickValueError("Word '{}' does not exist in the corpus.".format(word))

    def fit(self, X, y=None):
        # Instantiate the CountVectorizer
        try:
            vecs = CountVectorizer(
                vocabulary=self.words,
                lowercase=self.ignore_case,
                ngram_range=self.ngram_range,
                binary=True
            )
        except TypeError as e:
            raise YellowbrickValueError("Invalid parameter(s) passed to sklearn CountVectorizer: ", e)

        # Get the binary document counts for the target words
        self.doc_term_matrix_ = vecs.fit_transform(X)

        # Construct an index of terms to indices in the doc-term matrix
        self.docs_ = self.doc_term_matrix_.shape[0]
        self.vocab_ = vecs.get_feature_names_out()
        self._construct_term_index()

        # Compute the phi-coefficient for each pair of words
        self.plot_dim_ = len(self.words)
        self.correlation_matrix_ = np.zeros((self.plot_dim_, self.plot_dim_))
        for i, m in enumerate(self.words):
            for j, n in enumerate(self.words):
                self.correlation_matrix_[i, j] = self._compute_coefficient(m, n)

        self.draw(X)
        return self

    def draw(self, X):
        """
        Renders the correlation matrix as a heatmap.
        """
        # Use correlation matrix data for the heatmap
        wc_display = self.correlation_matrix_
        labels = self.words
        n_classes = len(labels)

        # Set up the dimensions of the pcolormesh
        X, Y = np.arange(n_classes + 1), np.arange(n_classes + 1)
        self.ax.set_ylim(bottom=0, top=wc_display.shape[0])
        self.ax.set_xlim(left=0, right=wc_display.shape[1])

        # Set the words as the tick labels on the plot. The Y-axis is sorted from top
        # to bottom, the X-axis is sorted from left to right
        xticklabels = labels
        yticklabels = labels[::-1]
        ticks = np.arange(n_classes) + 0.5
        self.ax.set(xticks=ticks, yticks=ticks)
        self.ax.set_xticklabels(xticklabels, rotation="vertical", fontsize=self.fontsize)
        self.ax.set_yticklabels(yticklabels, fontsize=self.fontsize)

        # Flip the Y-axis values so that they match the sorted labels
        wc_display = wc_display[::, ::-1]

        # Draw the labels in each heatmap cell
        for x in X[:-1]:
            for y in Y[:-1]:
                # Get the correlation value for the cell
                value = wc_display[x, y]
                svalue = "{:.2f}".format(value)

                # Get a compatible text color for the cell
                base_color = self.cmap(value / 2 + 0.5)
                text_color = find_text_color(base_color)

                # Draw the text at the center of the cell
                cx, cy = x + 0.5, y + 0.5
                self.ax.text(cx, cy, svalue, va="center", ha="center", color=text_color, fontsize=self.fontsize)

        # Draw the heatmap
        g = self.ax.pcolormesh(X, Y, wc_display, cmap=self.cmap, vmin=-1, vmax=1)

        # Add the color bar
        if self.colorbar:
            self.ax.figure.colorbar(g, ax=self.ax)

        return self.ax

    def finalize(self):
        self.set_title("Word Correlation Plot")
        self.fig.tight_layout()

##########################################################################
## Quick Method
##########################################################################

def word_correlation(
    words,
    corpus,
    ignore_case=True,
    ngram_range=(1, 1),
    ax=None,
    cmap="RdYlBu",
    show=True,
    colorbar=True,
    fontsize=None,
    **kwargs
):
    """Word Correlation

    Displays the binary correlation between the given words across the documents in a
    corpus.
    For a list of words with length n, this produces an n x n heatmap of correlation
    values in the range [-1, 1].

    Parameters
    """
    # Instantiate the visualizer
    visualizer = WordCorrelationPlot(
        words=words,
        lowercase=ignore_case,
        ngram_range=ngram_range,
        ax=ax,
        cmap=cmap,
        colorbar=colorbar,
        fontsize=fontsize,
        **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(corpus)

    # Draw the final visualization
    if show:
        visualizer.show()
    else:
        visualizer.finalize()

    # Return the visualizer
    return visualizer