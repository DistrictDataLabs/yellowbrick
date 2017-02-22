# yellowbrick.text.freqdist
# Implementations of frequency distributions for text visualization.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  2017-02-08 10:06
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: fredist.py [] rbilbro@districtdatalabs.com $

"""
Implementations of frequency distributions for text visualization
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

from operator import itemgetter

from yellowbrick.text.base import TextVisualizer
from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.style.colors import resolve_colors, get_color_cycle


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


class FreqDistVisualizer(TextVisualizer):
    """
    A frequency distribution tells us the frequency of each vocabulary
    item in the text. In general, it could count any kind of observable
    event. It is a distribution because it tells us how the total
    number of word tokens in the text are distributed across the
    vocabulary items.


    Parameters
    ----------
    ax : matplotlib axes
        The axes to plot the figure on.
    color : list or tuple of colors
        Specify color for bars
    N: integer
        Top N tokens to be plotted.
    kwargs : dict
        Pass any additional keyword arguments to the super class.

    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """
    def __init__(self, ax=None, color=None, N=50, **kwargs):
        """
        Initializes the base frequency distributions with many
        of the options required in order to make this
        visualization work.
        """
        super(FreqDistVisualizer, self).__init__(ax=ax, **kwargs)

        # Visualizer parameters
        self.N = 50

        # Visual Parameters
        self.color = color

    def freq_dist(self):
        """
        Called from the fit method, this method gets all the
        words from the corpus and their corresponding frequency
        counts.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """
        counts = np.asarray(self.docs.sum(axis=0)).ravel().tolist()
        self.word_freq = list(zip(self.features, counts))

    def get_counts(self):
        """
        Called from the fit method, this method sorts the words
        from the corpus with their corresponding frequency
        counts in reverse order.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """
        sorted_word_freq = sorted(self.word_freq,
                                  key=itemgetter(1), reverse=True)
        self.words, self.counts = list(zip(*sorted_word_freq))

    def fit(self, docs, features):
        """
        The fit method is the primary drawing input for the frequency
        distribution visualization. It requires vectorized lists of
        documents and a list of features, which are the actual words
        from the original corpus (needed to label the x-axis ticks).

        Parameters
        ----------
        docs : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features representing the corpus of
            vectorized documents.

        features : list
            List of corpus vocabulary words

        Text documents must be vectorized before passing to fit()
        """

        self.docs     = docs
        self.features = features

        self.freq_dist()
        self.get_counts()
        self.draw()

    def draw(self, **kwargs):
        """
        Called from the fit method, this method creates the canvas and
        draws the distribution plot on it.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """
        # Create the axis if it doesn't exist
        if self.ax is None: self.ax = plt.gca()

        # Plot the top 50 most frequent words
        y_pos = np.arange(self.N)
        self.ax.bar(y_pos, self.counts[:self.N], align='center', alpha=0.5)

        # Set the tick marks
        self.ax.set_xticks(y_pos)

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
            'Frequency distribution for top {} tokens'.format(self.N)
        )

        # Rotate tick marks to make words legible
        self.ax.set_xticklabels(self.words[:self.N], rotation=90)

        # Set the legend and the grid
        self.ax.legend(loc='best')
        self.ax.grid()
