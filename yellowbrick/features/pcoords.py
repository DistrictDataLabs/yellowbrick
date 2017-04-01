# yellowbrick.features.pcoords
# Implementations of parallel coordinates for feature analysis.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Mon Oct 03 21:46:06 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: pcoords.py [] benjamin@bengfort.com $

"""
Implementations of parallel coordinates for multi-dimensional feature
analysis. There are a variety of parallel coordinates from Andrews Curves to
coordinates that optimize column order.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

from yellowbrick.features.base import DataVisualizer
from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.style.colors import resolve_colors, get_color_cycle

##########################################################################
## Quick Methods
##########################################################################

def parallel_coordinates(X, y=None, ax=None, features=None, classes=None,
                         color=None, colormap=None, vlines=True,
                         vlines_kwds=None, **kwargs):
    """Displays each feature as a vertical axis and each instance as a line.

    This helper function is a quick wrapper to utilize the ParallelCoordinates
    Visualizer (Transformer) for one-off analysis.

    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features

    y : ndarray or Series of length n
        An array or series of target or class values

    ax : matplotlib axes
        The axes to plot the figure on.

    features : list of strings
        The names of the features or columns

    classes : list of strings
        The names of the classes in the target

    color : list or tuple of colors
        Specify the colors for each individual class

    colormap : string or matplotlib cmap
        Sequential colormap for continuous target

    vlines : bool
        Display the vertical azis lines

    vlines_kwds : dict
        Keyword arguments to draw the vlines

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the parallel coordinates were drawn on.
    """
    # Instantiate the visualizer
    visualizer = ParallelCoordinates(
        ax, features, classes, color, colormap, vlines, vlines_kwds, **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y, **kwargs)
    visualizer.transform(X)

    # Return the axes object on the visualizer
    return visualizer.ax


##########################################################################
## Static Parallel Coordinates Visualizer
##########################################################################

class ParallelCoordinates(DataVisualizer):
    """
    Parallel coordinates displays each feature as a vertical axis spaced
    evenly along the horizontal, and each instance as a line drawn between
    each individual axis.
    """

    def __init__(self, ax=None, features=None, classes=None, color=None,
                 colormap=None, vlines=True, vlines_kwds=None, **kwargs):
        """
        Initialize the base parallel coordinates with many of the options
        required in order to make the visualization work.

        Parameters
        ----------

        :param ax: the axis to plot the figure on.

        :param features: a list of feature names to use
            If a DataFrame is passed to fit and features is None, feature
            names are selected as the columns of the DataFrame.

        :param classes: a list of class names for the legend
            If classes is None and a y value is passed to fit then the classes
            are selected from the target vector.

        :param color: optional list or tuple of colors to colorize lines
            Use either color to colorize the lines on a per class basis or
            colormap to color them on a continuous scale.

        :param colormap: optional string or matplotlib cmap to colorize lines
            Use either color to colorize the lines on a per class basis or
            colormap to color them on a continuous scale.

        :param vlines: flag to determine vertical line display, default True

        :param vlines_kwds: options to style or display the vertical lines

        :param kwargs: keyword arguments passed to the super class.

        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.
        """
        super(ParallelCoordinates, self).__init__(
            ax, features, classes, color, colormap, **kwargs
        )

        # Visual Parameters
        self.show_vlines = vlines
        self.vlines_kwds = vlines_kwds or {
            'linewidth': 1, 'color': 'black'
        }

    def draw(self, X, y, **kwargs):
        """
        Called from the fit method, this method creates the parallel
        coordinates canvas and draws each instance and vertical lines on it.
        """
        # Get the shape of the data
        nrows, ncols = X.shape

        # Create the xticks for each column
        # TODO: Allow the user to specify this feature
        x = list(range(ncols))

        # Create the axis if it doesn't exist
        if self.ax is None: self.ax = plt.gca()

        # Create the colors
        # TODO: Allow both colormap, listed colors, and palette definition
        # TODO: Make this an independent function or property for override!
        # color_values = resolve_colors(
        #     num_colors=len(self.classes_), colormap=self.colormap, color=self.color
        # )
        color_values = get_color_cycle()
        colors = dict(zip(self.classes_, color_values))

        # Track which labels are already in the legend
        used_legends = set([])

        # TODO: Make this function compatible with DataFrames!
        # TODO: Make an independent function to allow addition of instances!
        for idx, row in enumerate(X):
            # TODO: How to map classmap to labels?
            label = y[idx] # Get the label for the row
            label = self.classes_[label]

            if label not in used_legends:
                used_legends.add(label)
                self.ax.plot(x, row, color=colors[label], label=label, **kwargs)
            else:
                self.ax.plot(x, row, color=colors[label], **kwargs)

        # Add the vertical lines
        # TODO: Make an independent function for override!
        if self.show_vlines:
            for idx in x:
                self.ax.axvline(idx, **self.vlines_kwds)

        # Set the limits
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(self.features_)
        self.ax.set_xlim(x[0], x[-1])

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
            'Parallel Coordinates for {} Features'.format(len(self.features_))
        )

        # Set the legend and the grid
        self.ax.legend(loc='best')
        self.ax.grid()
