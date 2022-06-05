# yellowbrick.text.correlation
# Implementation of word correlation for text visualization.
#
# Author:   Patrick Deziel
# Created:  Sun May 1 19:43:41 2022 -0600
#
# Copyright (C) 2022 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: correlation.py [b652fc9] deziel.patrick@gmail.com $

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

class WordCorrelationPlot(TextVisualizer):
    """
    Word correlation illustrates the extent to which words in a corpus appear in the
    same documents.

    WordCorrelationPlot visualizes the binary correlation between words across
    documents as a heatmap. The correlation is defined using the mean square
    contingency coefficient (phi-coefficient) between any two words m and n. The
    coefficient is a value between -1 and 1, inclusive. A value close to 1 or -1
    indicates strong positive or negative correlation between m and n, while a value
    close to 0 indicates little or no correlation. The constructor takes one required
    argument, which is the list of words or n-grams to be plotted.

    Parameters
    ----------
    words : list of str
        The list of words or n-grams to be plotted. The words must be present in the
        provided corpus on fit().

    ignore_case : bool, default: False
        If True, all words will be converted to lowercase before processing.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on.

    cmap : str or cmap, default: "RdYlBu"
        Colormap to use for the heatmap.

    colorbar : bool, default: True
        If True, a colorbar will be added to the heatmap.

    fontsize : int, default: None
        Font size to use for the labels on the axes.

    kwargs : dict
        Pass any additional keyword arguments to the super class.

    Attributes
    ----------
    self.doc_term_matrix_ : array of shape (n_docs, n_features)
        The computed sparse document-term matrix containing binary values indicating if
        a word is present in a document.

    self.num_docs_ : int
        The number of observed documents in the corpus.

    self.vocab_ : dict
        A dictionary mapping words to their indices in the document-term matrix.

    self.num_features_ : int
        The number of features (word labels) in the resulting plot.

    self.correlation_matrix_ : ndarray of shape (n_features, n_features)
        The computed matrix containing the phi-coefficients between all features.
    """
    def __init__(
        self,
        words,
        ignore_case=False,
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
        self.words = self._construct_terms(words, ignore_case)
        self.ngram_range = self._compute_ngram_range()

    def _construct_terms(self, words, ignore_case):
        """
        Constructs the list of terms to be plotted based on the provided words. This
        performs input checking and removes duplicates to produce a list of valid words
        for fitting.
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

    def _compute_ngram_range(self):
        """
        Computes the n-gram range to use for vectorization based on the provided words.
        This allows the user to specify multi-word terms for plotting.
        """
        ngrams = [len(word.split()) for word in self.words]
        return (min(ngrams), max(ngrams))

    def _compute_coefficient(self, m, n):
        """
        Computes the phi-coefficient for two words m and n, which is a correlation
        value between -1 and 1 inclusive.
        """
        m_col = self.doc_term_matrix_.getcol(self.vocab_[m])
        n_col = self.doc_term_matrix_.getcol(self.vocab_[n])
        both = m_col.multiply(n_col).sum()
        m_total = m_col.sum()
        n_total = n_col.sum()
        only_m = m_total - both
        only_n = n_total - both
        neither = self.num_docs_ - both - only_m - only_n
        return ((both * neither) - (only_m * only_n)) / np.sqrt(m_total * n_total * (self.num_docs_ - m_total) * (self.num_docs_ - n_total))

    def fit(self, X, y=None):
        """
        The fit method is the primary drawing input for the word correlation
        visualization.

        Parameters
        ----------
        X : list of str or generator
            Should be provided as a list of strings or a generator yielding strings
            that represent the documents in the corpus.
        
        y : None
            Labels are not used for the word correlation visualization.

        Returns
        -------
        self: instance
            Returns the instance of the transformer/visualizer.

        Attributes
        ----------
        self.doc_term_matrix_ : array of shape (n_docs, n_features)
            The computed sparse document-term matrix containing binary values
            indicating if a word is present in a document.

        self.num_docs_ : int
            The number of observed documents in the corpus.

        self.vocab_ : dict
            A dictionary mapping words to their indices in the document-term matrix.

        self.num_features_ : int
            The number of features (word labels) in the resulting plot.

        self.correlation_matrix_ : ndarray of shape (n_features, n_features)
            The computed matrix containing the phi-coefficients between all features.
        """

        # Instantiate the CountVectorizer
        vecs = CountVectorizer(
            vocabulary=self.words,
            lowercase=self.ignore_case,
            ngram_range=self.ngram_range,
            binary=True
        )

        # Get the binary document counts for the target words
        self.doc_term_matrix_ = vecs.fit_transform(X)
        self.num_docs_ = self.doc_term_matrix_.shape[0]
        self.vocab_ = vecs.vocabulary_

        # Verify that all target words exist in the corpus
        for word in self.words:
            if self.doc_term_matrix_.getcol(self.vocab_[word]).sum() == 0:
                raise YellowbrickValueError("Word '{}' does not exist in the corpus.".format(word))

        # Compute the phi-coefficient for each pair of words
        self.num_features_ = len(self.words)
        self.correlation_matrix_ = np.zeros((self.num_features_, self.num_features_))
        for i, m in enumerate(self.words):
            for j, n in enumerate(self.words):
                self.correlation_matrix_[i, j] = self._compute_coefficient(m, n)

        self.draw(X)
        return self

    def draw(self, X):
        """
        Called from the fit() method, this metod draws the heatmap on the figure using
        the computed correlation matrix.
        """

        # Use correlation matrix data for the heatmap
        wc_display = self.correlation_matrix_

        # Set up the dimensions of the pcolormesh
        X, Y = np.arange(self.num_features_ + 1), np.arange(self.num_features_ + 1)
        self.ax.set_ylim(bottom=0, top=wc_display.shape[0])
        self.ax.set_xlim(left=0, right=wc_display.shape[1])

        # Set the words as the tick labels on the plot. The Y-axis is sorted from top
        # to bottom, the X-axis is sorted from left to right.
        xticklabels = self.words
        yticklabels = self.words[::-1]
        ticks = np.arange(self.num_features_) + 0.5
        self.ax.set(xticks=ticks, yticks=ticks)
        self.ax.set_xticklabels(xticklabels, rotation="vertical", fontsize=self.fontsize)
        self.ax.set_yticklabels(yticklabels, fontsize=self.fontsize)

        # Flip the Y-axis values so that they match the sorted labels
        wc_display = np.flipud(wc_display)

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
                # Note: x and y coordinates are swapped to match the pcolormesh
                cx, cy = y + 0.5, x + 0.5
                self.ax.text(cx, cy, svalue, va="center", ha="center", color=text_color, fontsize=self.fontsize)

        # Draw the heatmap
        g = self.ax.pcolormesh(X, Y, wc_display, cmap=self.cmap, vmin=-1, vmax=1)

        # Add the color bar
        if self.colorbar:
            self.ax.figure.colorbar(g, ax=self.ax)

        return self.ax

    def finalize(self):
        """
        Prepares the figure for rendering by adding the title. This method is usually
        called from show() and not directly by the user.
        """
        self.set_title("Word Correlation Plot")
        self.fig.tight_layout()

##########################################################################
## Quick Method
##########################################################################

def word_correlation(
    words,
    corpus,
    ignore_case=True,
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
    ----------

    words : list of str
        The corpus words to display in the heatmap.
    corpus : list of str or generator
        The corpus as a list of documents or a generator yielding documents.

    ignore_case : bool, default: True
        If True, all words will be converted to lowercase before proessing.

    ax : matplotlib axes, default: None
        The axes to plot the figure on.

    cmap : str, default: "RdYlBu"
        Colormap to use for the heatmap.

    show : bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however
        you cannot call ``plt.savefig`` from this signature, nor
        ``clear_figure``. If False, simply calls ``finalize()``

    colorbar : bool, default: True
        If True, adds a colorbar to the figure.

    fontsize : int, default: None
        If not None, sets the font size of the labels.
    """
    # Instantiate the visualizer
    visualizer = WordCorrelationPlot(
        words=words,
        lowercase=ignore_case,
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