# yellowbrick.features.radviz
# Implements radviz for feature analysis.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Oct 07 13:18:00 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: radviz.py [0f4b236] benjamin@bengfort.com $

"""
Implements radviz for feature analysis.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.patches as patches

from yellowbrick.utils import is_dataframe
from yellowbrick.features.base import DataVisualizer
import yellowbrick.utils.nan_warnings as nan_warnings
from yellowbrick.style.colors import resolve_colors


##########################################################################
## Quick Methods
##########################################################################

def radviz(X, y=None, ax=None, features=None, classes=None,
           color=None, colormap=None, **kwargs):
    """
    Displays each feature as an axis around a circle surrounding a scatter
    plot whose points are each individual instance.

    This helper function is a quick wrapper to utilize the RadialVisualizer
    (Transformer) for one-off analysis.

    Parameters
    ----------

    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features

    y : ndarray or Series of length n
        An array or series of target or class values

    ax : matplotlib Axes, default: None
        The axes to plot the figure on.

    features : list of strings, default: None
        The names of the features or columns

    classes : list of strings, default: None
        The names of the classes in the target

    color : list or tuple of colors, default: None
        Specify the colors for each individual class

    colormap : string or matplotlib cmap, default: None
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

class RadialVisualizer(DataVisualizer):
    """
    RadViz is a multivariate data visualization algorithm that plots each
    axis uniformely around the circumference of a circle then plots points on
    the interior of the circle such that the point normalizes its values on
    the axes from the center to each arc.

    Parameters
    ----------

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    features : list, default: None
        a list of feature names to use
        If a DataFrame is passed to fit and features is None, feature
        names are selected as the columns of the DataFrame.

    classes : list, default: None
        a list of class names for the legend
        If classes is None and a y value is passed to fit then the classes
        are selected from the target vector.

    color : list or tuple, default: None
        optional list or tuple of colors to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    colormap : string or cmap, default: None
        optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------

    >>> visualizer = RadViz()
    >>> visualizer.fit(X, y)
    >>> visualizer.transform(X)
    >>> visualizer.poof()

    Notes
    -----
    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """

    def __init__(self, ax=None, features=None, classes=None, color=None,
                 colormap=None, **kwargs):
        super(RadialVisualizer, self).__init__(
            ax, features, classes, color, colormap, **kwargs
        )

    @staticmethod
    def normalize(X):
        """
        MinMax normalization to fit a matrix in the space [0,1] by column.
        """
        a = X.min(axis=0)
        b = X.max(axis=0)
        return (X - a[np.newaxis, :]) / ((b - a)[np.newaxis, :])

    def draw(self, X, y, **kwargs):
        """
        Called from the fit method, this method creates the radviz canvas and
        draws each instance as a class or target colored point, whose location
        is determined by the feature data set.
        """
        # Convert from dataframe
        if is_dataframe(X):
            X = X.as_matrix()

        # Clean out nans and warn that the user they aren't plotted
        nan_warnings.warn_if_nans_exist(X)
        X, y = nan_warnings.filter_missing(X, y)

        # Get the shape of the data
        nrows, ncols = X.shape

        # Set the axes limits
        self.ax.set_xlim([-1,1])
        self.ax.set_ylim([-1,1])

        # Create the colors
        # TODO: Allow both colormap, listed colors, and palette definition
        # TODO: Make this an independent function or property for override!
        color_values = resolve_colors(
            n_colors=len(self.classes_), colormap=self.colormap, colors=self.color
        )
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
        self.ax.add_patch(patches.Circle((0.0, 0.0), radius=1.0, facecolor='none', edgecolor='grey', linewidth=.5 ))

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
            'RadViz for {} Features'.format(len(self.features_))
        )

        # Remove the ticks from the graph
        self.ax.set_yticks([])
        self.ax.set_xticks([])

        # Add the legend
        self.ax.legend(loc='best')


# Alias for RadViz
RadViz = RadialVisualizer
