# yellowbrick.features.radviz
# Implements radviz for feature analysis.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Oct 07 13:18:00 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: radviz.py [] benjamin@bengfort.com $

"""
Implements radviz for feature analysis.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from yellowbrick.utils import is_dataframe
from yellowbrick.features.base import FeatureVisualizer
from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.style.colors import resolve_colors, get_color_cycle


##########################################################################
## Quick Methods
##########################################################################

def radviz(X, y=None, ax=None, features=None, classes=None,
           color=None, colormap=None, **kwargs):
    """Displays each feature as an axis around a circle surrounding a scatter
    plot whose points are each individual instance.

    This helper function is a quick wrapper to utilize the RadialVisualizer
    (Transformer) for one-off analysis.

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

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the parallel coordinates were drawn on.
    """
    # Instantiate the visualizer
    visualizer = RadialVisualizer(
        ax, features, classes, color, colormap, **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y, **kwargs)
    visualizer.transform(X)

    # Return the axes object on the visualizer
    return visualizer.ax


##########################################################################
## Static RadViz Visualizer
##########################################################################

class RadialVisualizer(FeatureVisualizer):
    """
    RadViz is a multivariate data visualization algorithm that plots each
    axis uniformely around the circumference of a circle then plots points on
    the interior of the circle such that the point normalizes its values on
    the axes from the center to each arc.
    """

    def __init__(self, ax=None, features=None, classes=None, color=None,
                 colormap=None, **kwargs):
        """
        Initialize the base radviz with many of the options required in order
        to make the visualization work.

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

        :param kwargs: keyword arguments passed to the super class.

        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.
        """
        super(RadialVisualizer, self).__init__(**kwargs)

        # The figure params
        # TODO: hoist to a higher level base class
        self.ax = ax

        # Data Parameters
        self.features_ = features
        self.classes_  = classes

        # Visual Parameters
        self.color = color
        self.colormap = colormap

    @staticmethod
    def normalize(X):
        """
        MinMax normalization to fit a matrix in the space [0,1] by column.
        """
        a = X.min(axis=0)
        b = X.max(axis=0)
        return (X - a[np.newaxis, :]) / ((b - a)[np.newaxis, :])

    def fit(self, X, y=None, **kwargs):
        """
        The fit method is the primary drawing input for the parallel coords
        visualization since it has both the X and y data required for the
        viz and the transform method does not.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs : dict
            Pass generic arguments to the drawing method

        Returns
        ------
        self : instance
            Returns the instance of the transformer/visualizer
        """
        # TODO: This class is identical to the Parallel Coordinates version,
        # so hoist this functionality to a higher level class that is extended
        # by both RadViz and ParallelCoordinates.

        # Get the shape of the data
        nrows, ncols = X.shape

        # Store the classes for the legend if they're None.
        if self.classes_ is None:
            # TODO: Is this the most efficient method?
            self.classes_ = [str(label) for label in set(y)]

        # Handle the feature names if they're None.
        if self.features_ is None:

            # If X is a data frame, get the columns off it.
            if is_dataframe(X):
                self.features_ = X.columns

            # Otherwise create numeric labels for each column.
            else:
                self.features_ = [
                    str(cdx) for cdx in range(ncols)
                ]

        # Draw the instances
        self.draw(X, y, **kwargs)

        # Fit always returns self.
        return self

    def draw(self, X, y, **kwargs):
        """
        Called from the fit method, this method creates the parallel
        coordinates canvas and draws each instance and vertical lines on it.
        """

        # Get the shape of the data
        nrows, ncols = X.shape

        # Create the axes if they don't exist
        if self.ax is None:
            self.ax = plt.gca(xlim=[-1,1], ylim=[-1,1])

        # Create the colors
        # TODO: Allow both colormap, listed colors, and palette definition
        # TODO: Make this an independent function or property for override!
        # color_values = resolve_colors(
        #     num_colors=len(self.classes_), colormap=self.colormap, color=self.color
        # )
        color_values = get_color_cycle()
        colors = dict(zip(self.classes_, color_values))

        # Create a data structure to hold scatter plot representations
        to_plot = {}
        for kls in self.classes_:
            to_plot[kls] = [[], []]

        # Compute the arcs around the circumference for each feature axis
        # TODO: make this an independent function for override
        s = np.array([
                (np.cos(t), np.sin(t))
                for t in [
                    2.0 * np.pi * (i / float(ncols))
                    for i in range(ncols)
                ]
            ])

        # Compute the locations of the scatter plot for each class
        # Normalize the data first to plot along the 0, 1 axis
        for i, row in enumerate(self.normalize(X)):
            row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)
            xy   = (s * row_).sum(axis=0) / row.sum()
            kls = self.classes_[y[i]]

            to_plot[kls][0].append(xy[0])
            to_plot[kls][1].append(xy[1])

        # Add the scatter plots from the to_plot function
        # TODO: store these plots to add more instances to later
        # TODO: make this a separate function
        for i, kls in enumerate(self.classes_):
            self.ax.scatter(to_plot[kls][0], to_plot[kls][1], color=colors[kls], label=str(kls), **kwargs)

        # Add the circular axis path
        # TODO: Make this a seperate function (along with labeling)
        self.ax.add_patch(patches.Circle((0.0, 0.0), radius=1.0, facecolor='none'))

        # Add the feature names
        for xy, name in zip(s, self.features_):
            # Add the patch indicating the location of the axis
            self.ax.add_patch(patches.Circle(xy, radius=0.025, facecolor='#777777'))

            # Add the feature names offset around the axis marker
            if xy[0] < 0.0 and xy[1] < 0.0:
                self.ax.text(xy[0] - 0.025, xy[1] - 0.025, name, ha='right', va='top', size='small')
            elif xy[0] < 0.0 and xy[1] >= 0.0:
                self.ax.text(xy[0] - 0.025, xy[1] + 0.025, name, ha='right', va='bottom', size='small')
            elif xy[0] >= 0.0 and xy[1] < 0.0:
                self.ax.text(xy[0] + 0.025, xy[1] - 0.025, name, ha='left', va='top', size='small')
            elif xy[0] >= 0.0 and xy[1] >= 0.0:
                self.ax.text(xy[0] + 0.025, xy[1] + 0.025, name, ha='left', va='bottom', size='small')

        self.ax.axis('equal')

    def poof(self, outpath=None, **kwargs):
        """
        Display the radial visualization

        Parameters
        ----------
        outpath: path or None
            Save the figure to disk or if None show in a window
        """
        if self.ax is None: return
        self.ax.legend(loc='best')

        if outpath is not None:
            plt.savefig(outpath, **kwargs)
        else:
            plt.show()

# Alias for RadViz
RadViz = RadialVisualizer
