
# yellowbrick.features.histogram
# Implementations of histogram with vertical lines to help with balanced binning.
#
# Author:   Juan L. Kehoe (juanluo2008@gmail.com)
# Author:   Prema Damodaran Roman (pdamo24@gmail.com)

# Created:  Tue Mar 13 19:50:54 2018 -0400
#
# Copyright (C) 2018 District Data Labs
# For license information, see LICENSE.txt
#
# ID: histogram.py

"""
Implements histogram with vertical lines to help with balanced binning.
"""

##########################################################################
## Imports
##########################################################################
import matplotlib.pyplot as plt
import numpy as np

from yellowbrick.features.base import FeatureVisualizer
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.utils import is_dataframe

##########################################################################
## Balanced Binning Reference
##########################################################################

class BalancedBinningReference(FeatureVisualizer):
    """
    BalancedBinningReference allows to generate a histogram with vertical lines
    showing the recommended value point to bin your data so they can be evenly
    distributed in each bin.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        This is inherited from FeatureVisualizer and is defined within
        BalancedBinningReference.
    feature : string, default: None
        The name of the X variable
        If a DataFrame is passed to fit and feature is None, feature
        is selected as the column of the DataFrame.  There must be only
        one column in the DataFrame.
    target : string, default: None
        The name of the Y variable
        Default: Counts
    bins : number of bins to generate the histogram
    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------
    >>> visualizer = BalancedBinningReference()
    >>> visualizer.fit(X)
    >>> visualizer.poof()


    Notes
    -----
    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """

    def __init__(self, ax=None, feature=None, target='Counts', bins=4, **kwargs):

        super(BalancedBinningReference, self).__init__(ax, **kwargs)

        self.feature = feature
        self.target = target
        self.bins = bins

    def draw(self, X, **kwargs):
        """
        Draws a histogram with the reference value for binning as vetical lines
        Parameters
        ----------
        X : ndarray or DataFrame of shape n x 1
            A matrix of n instances with 1 feature
        """

        # draw the histogram
        hist, bin_edges = np.histogram(X, bins=self.bins)
        self.ax.hist(X, bins=self.bins, color=kwargs.pop("color", "#6897bb"), **kwargs)

        # add vetical line with binning reference values
        plt.vlines(bin_edges,0,max(hist),colors=kwargs.pop("colors", "r"))

        #print the binning reference values
        print("The binning reference values are:", bin_edges)

    def fit(self, X, **kwargs):
        """
        Sets up X for the histogram and checks to
        ensure that X is of the correct data type
        Fit calls draw
        Parameters
        ----------
        X : ndarray or DataFrame of shape n x 1
            A matrix of n instances with 1 feature
        kwargs : dict
            keyword arguments passed to Scikit-Learn API.
        """

        #throw an error if X has more than 1 column
        if is_dataframe(X):
            nrows, ncols = X.shape

            if ncols > 1:
                raise YellowbrickValueError((
                    "X needs to be an ndarray or DataFrame with one feature, "
                    "please select one feature from the DataFrame"
                ))

        # Handle the feature name if it is None.
        if self.feature is None:

            # If X is a data frame, get the columns off it.
            if is_dataframe(X):
                self.feature = X.columns

            else:
                self.feature = ['X']

        self.draw(X)
        return self


    def poof(self, **kwargs):
        """
        Creates the labels for the feature and target variables
        """

        self.ax.set_xlabel(self.feature)
        self.ax.set_ylabel(self.target)
        self.finalize(**kwargs)

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.
        Parameters
        ----------
        kwargs: generic keyword arguments.
        """

        for tk in self.ax.get_xticklabels():
            tk.set_visible(True)
            
        for tk in self.ax.get_yticklabels():
            tk.set_visible(True)
        
        
##########################################################################
## Quick Method
##########################################################################
        
def balanced_binning_reference(ax=None, feature=None, target='Counts', bins=4, **kwargs):
    
    """
    BalancedBinningReference allows to generate a histogram with vertical lines
    showing the recommended value point to bin your data so they can be evenly
    distributed in each bin.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        This is inherited from FeatureVisualizer and is defined within
        BalancedBinningReference.
    feature : string, default: None
        The name of the X variable
        If a DataFrame is passed to fit and feature is None, feature
        is selected as the column of the DataFrame.  There must be only
        one column in the DataFrame.
    target : string, default: None
        The name of the Y variable
        Default: Counts
    bins : number of bins to generate the histogram
    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    """
    
    # Initialize the visualizer
    visualizer = balanced_binning_reference(ax=ax, X, y, bins=bins)
    
    # Fit and poof the visualizer
    visualizer.fit(X)
    visualizer.poof()
    
    

    
    
    

