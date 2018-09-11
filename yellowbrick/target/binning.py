
# yellowbrick.target.binning
# Implementations of histogram with vertical lines to help with balanced binning.
#
# Author:   Juan L. Kehoe (juanluo2008@gmail.com)
# Author:   Prema Damodaran Roman (pdamo24@gmail.com)

# Created:  Tue Mar 13 19:50:54 2018 -0400
#
# Copyright (C) 2018 District Data Labs
# For license information, see LICENSE.txt
#
# ID: binning.py

"""
Implements histogram with vertical lines to help with balanced binning.
"""

##########################################################################
## Imports
##########################################################################
import matplotlib.pyplot as plt
import numpy as np

from .base import TargetVisualizer
from yellowbrick.exceptions import YellowbrickValueError

##########################################################################
## Balanced Binning Reference
##########################################################################

class BalancedBinningReference(TargetVisualizer):
    """
    BalancedBinningReference generates a histogram with vertical lines
    showing the recommended value point to bin your data so they can be evenly
    distributed in each bin.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        This is inherited from FeatureVisualizer and is defined within
        ``BalancedBinningReference``.

    target : string, default: "Frequency"
        The name of the ``y`` variable

    bins : number of bins to generate the histogram, default: 4

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.
        
    Attributes
    ----------
    bin_edges : binning reference values

    Examples
    --------
    >>> visualizer = BalancedBinningReference()
    >>> visualizer.fit(y)
    >>> visualizer.poof()


    Notes
    -----
    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """

    def __init__(self, ax=None, target=None, bins=4, **kwargs):

        super(BalancedBinningReference, self).__init__(ax, **kwargs)

        self.target = target
        self.bins = bins

    def draw(self, y, **kwargs):
        """
        Draws a histogram with the reference value for binning as vertical
        lines.

        Parameters
        ----------
        y : an array of one dimension or a pandas Series
        """

        # draw the histogram
        hist, bin_edges = np.histogram(y, bins=self.bins)
        self.bin_edges_ = bin_edges
        self.ax.hist(y, bins=self.bins, color=kwargs.pop("color", "#6897bb"), **kwargs)

        # add vetical line with binning reference values
        plt.vlines(bin_edges,0,max(hist),colors=kwargs.pop("colors", "r"))

    def fit(self, y, **kwargs):
        """
        Sets up y for the histogram and checks to
        ensure that ``y`` is of the correct data type.
        Fit calls draw.

        Parameters
        ----------
        y : an array of one dimension or a pandas Series

        kwargs : dict
            keyword arguments passed to scikit-learn API.

        """

        #throw an error if y has more than 1 column
        if y.ndim > 1:
            raise YellowbrickValueError("y needs to be an array or Series with one dimension") 

        # Handle the target name if it is None.
        if self.target is None:
            self.target = 'Frequency'

        self.draw(y)
        return self


    def poof(self, **kwargs):
        """
        Creates the labels for the feature and target variables.
        """

        self.ax.set_xlabel(self.target)
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
        
def balanced_binning_reference(y, ax=None, target='Frequency', bins=4, **kwargs):
    
    """
    BalancedBinningReference generates a histogram with vertical lines
    showing the recommended value point to bin your data so they can be evenly
    distributed in each bin.

    Parameters
    ----------
    y : an array of one dimension or a pandas Series
    
    ax : matplotlib Axes, default: None
        This is inherited from FeatureVisualizer and is defined within
        ``BalancedBinningReference``.

    target : string, default: "Frequency"
        The name of the ``y`` variable

    bins : number of bins to generate the histogram, default: 4

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    """

    # Initialize the visualizer
    visualizer = BalancedBinningReference(ax=ax, bins=bins, target=target, **kwargs)
    
    # Fit and poof the visualizer
    visualizer.fit(y)
    visualizer.poof()
    
    

    
    
    

