# yellowbrick.text.freqdist
# Implementations of frequency distributions for text visualization.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  2017-02-08 10:06
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: freqdist.py [67b2740] rebecca.bilbro@bytecubed.com $

"""
Implementations of frequency distributions for text visualization
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from operator import itemgetter

from yellowbrick.text.base import TextVisualizer
from yellowbrick.exceptions import YellowbrickValueError


##########################################################################
## Quick Method
##########################################################################

def freqdist(X, y=None, ax=None, color=None, N=50, **kwargs):
    """Displays frequency distribution plot for text.

    This helper function is a quick wrapper to utilize the FreqDist
    Visualizer (Transformer) for one-off analysis.

    Parameters
    ----------

    X: ndarray or DataFrame of shape n x m
        A matrix of n instances with m features. In the case of text,
        X is a list of list of already preprocessed words

    y: ndarray or Series of length n
        An array or series of target or class values

    ax: matplotlib axes
        The axes to plot the figure on.

    color: string
        Specify color for barchart

    N: integer
        Top N tokens to be plotted.

    kwargs: dict
        Keyword arguments passed to the super class.

    Returns
    -------
    ax: matplotlib axes
        Returns the axes that the plot was drawn on.
    """
    # Instantiate the visualizer
    visualizer = FreqDistVisualizer(
        ax, X, color, **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y, **kwargs)
    visualizer.transform(X)

    # Return the axes object on the visualizer
    return visualizer.ax


class FrequencyVisualizer(TextVisualizer):
    """
    A frequency distribution tells us the frequency of each vocabulary
    item in the text. In general, it could count any kind of observable
    event. It is a distribution because it tells us how the total
    number of word tokens in the text are distributed across the
    vocabulary items.


    Parameters
    ----------
    features : list, default: None
        The list of feature names from the vectorizer, ordered by index. E.g.
        a lexicon that specifies the unique vocabulary of the corpus. This
        can be typically fetched using the ``get_feature_names()`` method of
        the transformer in Scikit-Learn.

    ax : matplotlib axes, default: None
        The axes to plot the figure on.

    n: integer, default: 50
        Top N tokens to be plotted.

    orient : 'h' or 'v', default: 'h'
        Specifies a horizontal or vertical bar chart.

    color : list or tuple of colors
        Specify color for bars

    kwargs : dict
        Pass any additional keyword arguments to the super class.

    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """

    def __init__(self, features, ax=None, n=50, orient='h', color=None, **kwargs):
        super(FreqDistVisualizer, self).__init__(ax=ax, **kwargs)

        # Check that the orient is correct
        orient = orient.lower().strip()
        if orient not in {'h', 'v'}:
            raise YellowbrickValueError(
                "Orientation must be 'h' or 'v'"
            )

        # Visualizer parameters
        self.N = n
        self.features = features

        # Visual arguments
        self.color = color
        self.orient = orient

    def count(self, X):
        """
        Called from the fit method, this method gets all the
        words from the corpus and their corresponding frequency
        counts.

        Parameters
        ----------

        X : ndarray or masked ndarray
            Pass in the matrix of vectorized documents, can be masked in
            order to sum the word frequencies for only a subset of documents.

        Returns
        -------

        counts : array
            A vector containing the counts of all words in X (columns)

        """
        # Sum on axis 0 (by columns), each column is a word
        # Convert the matrix to an array
        # Squeeze to remove the 1 dimension objects (like ravel)
        return np.squeeze(np.asarray(X.sum(axis=0)))

    def fit(self, X, y=None):
        """
        The fit method is the primary drawing input for the frequency
        distribution visualization. It requires vectorized lists of
        documents and a list of features, which are the actual words
        from the original corpus (needed to label the x-axis ticks).

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features representing the corpus
            of frequency vectorized documents.

        y : ndarray or DataFrame of shape n
            Labels for the documents for conditional frequency distribution.

        .. note:: Text documents must be vectorized before ``fit()``.
        """

        # Compute the conditional word frequency
        if y is not None:
            # Fit the frequencies
            self.conditional_freqdist_ = {}

            # Conditional frequency distribution
            self.classes_ = [str(label) for label in set(y)]
            for label in self.classes_:
                self.conditional_freqdist_[label] = self.count(X[y == label])
        else:
            # No conditional frequencies
            self.conditional_freqdist_ = None

        # Frequency distribution of entire corpus.
        self.freqdist_ = self.count(X)
        self.sorted_ = self.freqdist_.argsort()[::-1] # Descending order

        # Compute the number of words, vocab, and hapaxes
        self.vocab_ = self.freqdist_.shape[0]
        self.words_ = self.freqdist_.sum()
        self.hapaxes_ = sum(1 for c in self.freqdist_ if c == 1)

        # Draw and ensure that we return self
        self.draw()
        return self

    def draw(self, **kwargs):
        """
        Called from the fit method, this method creates the canvas and
        draws the distribution plot on it.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """
        # Prepare the data
        bins  = np.arange(self.N)
        words = [self.features[i] for i in self.sorted_[:self.N]]
        freqs = {}

        # Set up the bar plots
        if self.conditional_freqdist_:
            for label, values in sorted(self.conditional_freqdist_.items(), key=itemgetter(0)):
                freqs[label] = [
                    values[i] for i in self.sorted_[:self.N]
                ]
        else:
            freqs['corpus'] = [
                self.freqdist_[i] for i in self.sorted_[:self.N]
            ]

        # Draw a horizontal barplot
        if self.orient == 'h':
            # Add the barchart, stacking if necessary
            for label, freq in freqs.items():
                self.ax.barh(bins, freq, label=label, align='center')

            # Set the y ticks to the words
            self.ax.set_yticks(bins)
            self.ax.set_yticklabels(words)

            # Order the features from top to bottom on the y axis
            self.ax.invert_yaxis()

            # Turn off y grid lines and turn on x grid lines
            self.ax.yaxis.grid(False)
            self.ax.xaxis.grid(True)

        # Draw a vertical barplot
        elif self.orient == 'v':
            # Add the barchart, stacking if necessary
            for label, freq in freqs.items():
                self.ax.bar(bins, freq, label=label, align='edge')

            # Set the y ticks to the words
            self.ax.set_xticks(bins)
            self.ax.set_xticklabels(words, rotation=90)

            # Turn off x grid lines and turn on y grid lines
            self.ax.yaxis.grid(True)
            self.ax.xaxis.grid(False)

        # Unknown state
        else:
            raise YellowbrickValueError(
                "Orientation must be 'h' or 'v'"
            )

    def finalize(self, **kwargs):
        """
        The finalize method executes any subclass-specific axes
        finalization steps. The user calls poof & poof calls finalize.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """
        # Set the title
        self.set_title(
            'Frequency Distribution of Top {} tokens'.format(self.N)
        )

        # Create the vocab, count, and hapaxes labels
        infolabel = "vocab: {:,}\nwords: {:,}\nhapax: {:,}".format(
            self.vocab_, self.words_, self.hapaxes_
        )

        self.ax.text(0.68, 0.97, infolabel, transform=self.ax.transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox={'boxstyle':'round', 'facecolor':'white', 'alpha':.8})

        # Set the legend and the grid
        self.ax.legend(loc='upper right', frameon=True)


# Backwards compatibility alias
FreqDistVisualizer = FrequencyVisualizer
