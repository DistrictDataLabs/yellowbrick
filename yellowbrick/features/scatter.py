# yellowbrick.features.scatter
# Implements a 2d scatter plot for feature analysis.
#
# Author:   Nathan Danielsen <nathan.danielsen@gmail.com>
# Created:  Fri Feb 26 19:40:00 2017 -0400
#
# For license information, see LICENSE.txt
#
# ID: scatter.py [] nathan.danielsen@gmail.com $

"""
Implements a 2D scatter plot for feature analysis.
"""

##########################################################################
## Imports
##########################################################################
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from yellowbrick.features.base import DataVisualizer
from yellowbrick.utils import is_dataframe
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.style.colors import resolve_colors, get_color_cycle


##########################################################################
## Quick Methods
##########################################################################

def scatterviz(X, y=None, ax=None, features=None, classes=None,
           color=None, colormap=None, markers=None, **kwargs):
    """Displays a bivariate scatter plot.

    This helper function is a quick wrapper to utilize the ScatterVisualizer
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
        The names of two features or columns.
        More than that will raise an error.

    classes : list of strings
        The names of the classes in the target

    color : list or tuple of colors
        Specify the colors for each individual class

    colormap : string or matplotlib cmap
        Sequential colormap for continuous target

    :param markers: iterable of strings
        Matplotlib style markers for points on the scatter plot points

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the parallel coordinates were drawn on.
    """
    # Instantiate the visualizer
    visualizer = ScatterVisualizer(
        ax, features, classes, color, colormap, markers, **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y, **kwargs)
    visualizer.transform(X)

    # Return the axes object on the visualizer
    return visualizer.ax


##########################################################################
## Static ScatterVisualizer Visualizer
##########################################################################

class ScatterVisualizer(DataVisualizer):
    """
    ScatterVisualizer is a bivariate feature data visualization algorithm that
    plots using the Cartesian coordinates of each point.
    """

    def __init__(self, ax=None, features=None, classes=None, color=None,
                 colormap=None, markers=None, **kwargs):
        """
        Initialize the base scatter with many of the options required in order
        to make the visualization work.

        Parameters
        ----------

        :param ax: the axis to plot the figure on.

        :param features: a list of feature names to use
            List of features that correspond to the columns in the array.
            More than two feature names or coloumns will raise an error. If
            a DataFrame is passed to fit and features is None, feature
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

        :param markers: iterable of strings
            Matplotlib style markers for points on the scatter plot points

        :param kwargs: keyword arguments passed to the super class.

        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.
        """
        super(ScatterVisualizer, self).__init__(
            ax, features, classes, color, colormap, **kwargs
        )
        self.markers = itertools.cycle(kwargs.pop('markers', (',', '+', 'o', '*', 'v', 'h', 'd') ))

        # Ensure with init that features doesn't have more than two features
        if features is not None:
            if len(features) != 2:
                raise YellowbrickValueError('ScatterVisualizer only accepts two features')

    def fit(self, X, y=None, **kwargs):
        """
        The fit method is the primary drawing input for the parallel coords
        visualization since it has both the X and y data required for the
        viz and the transform method does not.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with 2 features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs : dict
            Pass generic arguments to the drawing method

        Returns
        -------
        self : instance
            Returns the instance of the transformer/visualizer
        """
        nrows, ncols = X.shape

        if ncols == 2:
            X_two_cols = X
            if self.features_ is None:
                self.features_ = [0, 1]

        # Handle the feature names if they're None.
        elif self.features_ is not None and is_dataframe(X):
            X_two_cols = X[self.features_].as_matrix()
        else:
            raise YellowbrickValueError("""
                ScatterVisualizer only accepts two features, please
                explicitly set these two features in the init kwargs or
                pass a matrix/ dataframe in with only two columns."""
            )

        # Store the classes for the legend if they're None.
        if self.classes_ is None:
            # TODO: Is this the most efficient method?
            self.classes_ = [str(label) for label in set(y)]

        # Draw the instances
        self.draw(X_two_cols, y, **kwargs)

        # Fit always returns self.
        return self

    def draw(self, X, y, **kwargs):
        """
        Called from the fit method, this method creates a scatter plot that draws
        each instance as a class or target colored point, whose location
        is determined by the feature data set.
        """
        # Get the shape of the data
        nrows, ncols = X.shape

        # Create the axes if they don't exist
        if self.ax is None:
                self.ax = plt.gca(xlim=[-1,1], ylim=[-1,1])

        color_values = get_color_cycle()
        colors = dict(zip(self.classes_, color_values))

        # Create a data structure to hold the scatter plot representations
        to_plot = {}
        for kls in self.classes_:
            to_plot[kls] = [[], []]

        # Add each row of the data set to to_plot for plotting
        # TODO: make this an independent function for override
        for i, row in enumerate(X):
            row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)
            x_, y_   = row_[0], row_[1]
            kls = self.classes_[y[i]]

            to_plot[kls][0].append(x_)
            to_plot[kls][1].append(y_)

        # Add the scatter plots from the to_plot function
        # TODO: store these plots to add more instances to later
        # TODO: make this a separate function
        for i, kls in enumerate(self.classes_):
            self.ax.scatter(to_plot[kls][0], to_plot[kls][1], marker=next(self.markers), color=colors[kls], label=str(kls), **kwargs)

        self.ax.axis('equal')

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """
        # Divide out the two features
        feature_one, feature_two = self.features_

        # Set the title
        self.set_title(
            'Scatter Plot: {0} vs {1}'.format(str(feature_one), str(feature_two))
        )
        # Add the legend
        self.ax.legend(loc='best')
        self.ax.set_xlabel(str(feature_one))
        self.ax.set_ylabel(str(feature_two))


# Alias for ScatterViz
ScatterViz = ScatterVisualizer
