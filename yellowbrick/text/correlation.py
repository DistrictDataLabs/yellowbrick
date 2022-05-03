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
from yellowbrick.text.base import TextVisualizer
from yellowbrick.style.palettes import color_sequence
from yellowbrick.exceptions import YellowbrickValueError

##########################################################################
## Word Correlation Plot Visualizer
##########################################################################

CMAP_UNDERCOLOR = "w"
CMAP_OVERCOLOR = "#2a7d4f"

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
        self.cmap.set_under(color=CMAP_UNDERCOLOR)
        self.cmap.set_over(color=CMAP_OVERCOLOR)

        # Estimator parameters

        self.words = words

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

    def _construct_terms(self):
        """
        Constructs the list of terms for the plot.
        """
        self.terms_ = []
        for word in self.words:
            word = word.lower()
            if word not in self.vocab_:
                raise YellowbrickValueError(("The word '{}' does not exist in the corpus").format(word))
            self.terms_.append(word)

    def _construct_term_index(self):
        """
        Constructs a dictionary mapping terms to indices in the sparse doc-term matrix.
        """
        self._construct_terms()
        self.term_index_ = {}
        for i, feature in enumerate(self.vocab_):
            if feature not in self.term_index_ and feature in self.terms_:
                self.term_index_[feature] = i
                if len(self.term_index_) == len(self.terms_):
                    break

    def fit(self, X, y=None):
        # Construct term list
        vecs = CountVectorizer(binary=True)
        self.doc_term_matrix_ = vecs.fit_transform(X)
        self.docs_ = self.doc_term_matrix_.shape[0]
        self.vocab_ = vecs.get_feature_names_out()
        self._construct_term_index()

        # Compute the phi-coefficient for each pair of words
        self.plot_dim_ = len(self.terms_)
        self.correlation_matrix_ = np.zeros((self.plot_dim_, self.plot_dim_))
        for i, m in enumerate(self.terms_):
            for j, n in enumerate(self.terms_):
                self.correlation_matrix_[i, j] = self._compute_coefficient(m, n)

        self.draw(X)
        return self

    def draw(self, X):
        """
        Renders the correlation matrix as a heatmap.
        """
        # Use correlation matrix data for the heatmap
        wc_display = self.correlation_matrix_
        labels = self.terms_
        n_classes = len(labels)

        # Set up the dimensions of the pcolormesh
        X, Y = np.arange(n_classes + 1), np.arange(n_classes + 1)
        self.ax.set_ylim(bottom=0, top=wc_display.shape[0])
        self.ax.set_xlim(left=0, right=wc_display.shape[1])

        # Set the words as the tick labels on the plot
        xticklabels = labels
        yticklabels = labels
        ticks = np.arange(n_classes) + 0.5
        self.ax.set(xticks=ticks, yticks=ticks)
        self.ax.set_xticklabels(xticklabels, rotation="vertical", fontsize=self.fontsize)
        self.ax.set_yticklabels(yticklabels, fontsize=self.fontsize)

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