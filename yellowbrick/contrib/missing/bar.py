# yellowbrick.contrib.missing.bar
# Missing Values Bar Visualizer
#
# Author:  Nathan Danielsen <nathan.danielsen@gmail.com>
# Created: Fri Mar 29 5:17:36 2018 -0500
#
# Copyright (C) 2018 District Data Labs
# For license information, see LICENSE.txt
#
# ID: bar.py [] nathan.danielsen@gmail.com.com $

"""
Bar visualizer of missing values by column.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from .base import MissingDataVisualizer

##########################################################################
## MissingValues Visualizer
##########################################################################


class MissingValuesBar(MissingDataVisualizer):
    """The MissingValues Bar visualizer creates a bar graph that lists the total
    count of missing values for each selected feature column.
    """

    def __init__(self, width=0.5, color='black', **kwargs):
        """
        """
        super(MissingValuesBar, self).__init__(**kwargs)
        self.width = width  # the width of the bars
        self.color = color  # the color of the bars

    def get_nan_col_counts(self, **kwargs):
        # where matrix is contains strings
        if np.issubdtype(self.X.dtype, np.string_) or np.issubdtype(self.X.dtype, np.unicode_):
            mask = np.where( self.X == '' )
            nan_matrix = np.zeros(self.X.shape)
            nan_matrix[mask] = np.nan

        else:
            nan_matrix = self.X.astype(np.float)

        nan_col_counts =  [np.count_nonzero(np.isnan(col)) for col in nan_matrix.T]
        return nan_col_counts

    def draw(self, X, y, **kwargs):
        """Called from the fit method, this method generated a bar plot.
        """
        nan_col_counts = self.get_nan_col_counts()

        # the x locations for the groups
        self.ind = np.arange(len(self.features_))

        self.ax.barh(self.ind - self.width / 2, nan_col_counts, self.width,
                        color=self.color)

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
            'Count of Missing Values by Column'
        )
        tick_locations = np.arange(len(self.features_))  # the x locations for the groups
        self.ax.set_yticks(tick_locations)
        self.ax.set_yticklabels(self.get_feature_names())
        # Remove the ticks from the graph
        self.ax.set_xlabel('Count')

        self.ax.legend()

##########################################################################
## Quick Method
##########################################################################

def missing_bar(X, y=None, ax=None, features=None, width=0.5, color='black', **kwargs):
    """The MissingValues Bar visualizer creates a bar graph that lists the total
    count of missing values for each selected feature column.
    """
    # Instantiate the visualizer
    visualizer = MissingValuesBar(
        ax=ax, features=features, width=width, color=color, **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y)
    visualizer.transform(X)
    visualizer.poof()

    # Return the axes object on the visualizer
    return visualizer.ax
