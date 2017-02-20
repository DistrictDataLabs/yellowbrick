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

from yellowbrick.text.base import TextVisualizer
from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.style.colors import resolve_colors, get_color_cycle


##########################################################################
## Quick Method
##########################################################################

def freqdist(X, y=None, ax=None, color=None, cumulative=False,
             N=50, **kwargs):
    """Displays frequency distribution plot for text.

    This helper function is a quick wrapper to utilize the FreqDist
    Visualizer (Transformer) for one-off analysis.

    Parameters
    ----------
    :param X: ndarray or DataFrame of shape n x m
        A matrix of n instances with m features. In the case of text,
        X is a list of list of already preprocessed words

    :param y: ndarray or Series of length n
        An array or series of target or class values

    :param ax: matplotlib axes
        The axes to plot the figure on.

    :param features: list of strings
        The names of the features or columns

    :param classes: list of strings
        The names of the classes in the target

    :param color: list or tuple of colors
        Specify the colors for each individual class

    :param N: integer
        Top N tokens to be plotted.

    :param cumulative: Boolean
        If True, plots the cumulative frequency distribution

    :param kwargs: dictionary
        Keyword arguments passed to the super class.

    Returns
    -------
    :param ax: matplotlib axes
        Returns the axes that the parallel coordinates were drawn on.
    """
    # Instantiate the visualizer
    visualizer = FreqDist(
        ax, X, classes, color, cumulative, N, **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y, **kwargs)
    visualizer.transform(X)

    # Return the axes object on the visualizer
    return visualizer.ax


class FreqDist(TextVisualizer):
    """
    A frequency distribution tells us the frequency of each vocabulary
    item in the text. In general, it could count any kind of observable
    event. It is a distribution because it tells us how the total
    number of word tokens in the text are distributed across the
    vocabulary items.
    """
    def __init__(self, ax=None, color=None, cumulative=False,
                 N=50, **kwargs):
        """
        Initialize the base frequency distributions with many of the options
        required in order to make the visualization work.

        Parameters
        ----------

        :param ax: the axis to plot the figure on.

        :param color: optional list or tuple of colors to colorize lines

        :param cumulative: Boolean
            If True, plots the cumulative frequency distribution

        :param N: integer
            Top N tokens to be plotted.

        :param kwargs: dictionary
            Keyword arguments passed to the super class.

        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.
        """
        super(FreqDist, self).__init__(
            ax, color, cumulative, N, **kwargs
        )

    # @staticmethod
    # def normalize(X):
    #     """
    #     Normalize the text (lower case, remove stopwords and punctuation)
    #     """
    #     # TODO: implement me
    #     pass
    #
    # @staticmethod
    # def get_interesting_words(X, N=50):
    #     """
    #     Call normalize and get the top 50 most interesting words
    #     """
    #     # TODO: implement me
    #     pass

    def draw(self, X, y, **kwargs):
        """
        Called from the fit method, this method creates the canvas and
        draws the distribution plot on it.
        """
        # TODO: implement me
        pass

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """
        # Set the title
        self.set_title(
            'Frequency distribution for top {} tokens'.format(self.N)
        )

        # Set the legend and the grid
        self.ax.legend(loc='best')
        self.ax.grid()
