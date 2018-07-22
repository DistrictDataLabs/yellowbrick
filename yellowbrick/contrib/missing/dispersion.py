# yellowbrick.contrib.missing.dispersion
# Missing Values Dispersion Visualizer
#
# Author:  Nathan Danielsen <nathan.danielsen@gmail.com>
# Created: Fri Mar 29 5:17:36 2018 -0500
#
# Copyright (C) 2018 District Data Labs
# For license information, see LICENSE.txt
#
# ID: dispersion.py [] nathan.danielsen@gmail.com.com $

"""
Dispersion visualizer for locations of missing values by column against index position.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from .base import MissingDataVisualizer
from yellowbrick.style.palettes import color_palette

##########################################################################
## MissingValues Visualizer
##########################################################################

class MissingValuesDispersion(MissingDataVisualizer):
    """
    The Missing Values Dispersion visualizer shows the locations of missing (nan)
    values in the feature dataset against the index column.
    """
    # TODO - map missing values against another user selected column such as a
    # datetime column.

    def __init__(self, alpha=0.5, marker="|", classes=None, **kwargs):
        """
        """

        super(MissingValuesDispersion, self).__init__(**kwargs)
        self.alpha = alpha
        self.marker = marker

        self.classes_ = classes

        # Convert to array if necessary to match estimator.classes_
        if self.classes_ is not None:
            self.classes_ = np.array(classes)

        # Set up classifier score visualization properties
        if self.classes_ is not None:
            n_colors = len(self.classes_)
        else:
            n_colors = None

        self.colors = color_palette(kwargs.pop('colors', None), n_colors)


    def get_nan_locs(self, **kwargs):
        """Gets the locations of nans in feature data and returns
        the coordinates in the matrix
        """
        if np.issubdtype(self.X.dtype, np.string_) or np.issubdtype(self.X.dtype, np.unicode_):
            mask = np.where( self.X == '' )
            nan_matrix = np.zeros(self.X.shape)
            nan_matrix[mask] = np.nan

        else:
            nan_matrix = self.X.astype(float)

        if self.y is None:
            return np.argwhere(np.isnan(nan_matrix))
        else:
            nan_locs = []
            for target_value in np.unique(self.y):
                indices = np.argwhere(self.y == target_value)
                target_matrix = nan_matrix[indices.flatten()]
                nan_target_locs = np.argwhere(np.isnan(target_matrix))
                nan_locs.append((target_value, nan_target_locs))

            return nan_locs

    def draw(self, X, y, **kwargs):
        """Called from the fit method, this method creates a scatter plot that
        draws each instance as a class or target colored point, whose location
        is determined by the feature data set.

        If y is not None, then it draws a scatter plot where each class is in a
        different color.
        """
        nan_locs = self.get_nan_locs()
        if y is None:
            x_, y_ = list(zip(*nan_locs))
            self.ax.scatter(x_, y_, alpha=self.alpha, marker=self.marker)
        else:
            self.draw_multi_dispersion_chart(nan_locs)

    def draw_multi_dispersion_chart(self, nan_locs):
        """Draws a multi dimensional dispersion chart, each color corresponds
        to a different target variable.
        """
        for index, nan_values in enumerate(nan_locs):
            label, nan_locations = nan_values

            # if features passed in then, label as such
            if self.classes_ is not None:
                label = self.classes_[index]

            color = self.colors[index]

            x_, y_ = list(zip(*nan_locations))
            self.ax.scatter(x_, y_, alpha=self.alpha, marker=self.marker, color=color, label=label)

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
            'Dispersion of Missing Values by Feature'
        )
        tick_locations = np.arange(len(self.features_))  # the x locations for the groups
        # Remove the ticks from the graph
        self.ax.set_xlabel('Position by index')
        self.ax.set_yticks(tick_locations)
        self.ax.set_yticklabels(self.get_feature_names())
        # Add the legend
        legend = self.ax.legend(loc='lower right', facecolor="grey")
        legend.get_frame().set_edgecolor('b')


##########################################################################
## Quick Method
##########################################################################

def missing_dispersion(X, y=None, ax=None, features=None, alpha=0.5, marker="|", **kwargs):
    """
    The Missing Values Dispersion visualizer shows the locations of missing (nan)
    values in the feature dataset against the index column.
    """
    # Instantiate the visualizer
    visualizer = MissingValuesDispersion(
        ax=ax, features=features, alpha=alpha, marker=marker, **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y)
    visualizer.transform(X)
    visualizer.poof()

    # Return the axes object on the visualizer
    return visualizer.ax
