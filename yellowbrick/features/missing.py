# yellowbrick.features.missing
# Feature importance visualizer
#
# Author:  Nathan Danielsen <nathan.danielsen@gmail.com>
# Created: Fri Mar 24 8:17:36 2018 -0500
#
# Copyright (C) 2018 District Data Labs
# For license information, see LICENSE.txt
#
# ID: missing.py [] nathan.danielsen@gmail.com.com $

"""
Implementation of missing values visualizers

To Include:
- Bar
- Density Matrix (by time, specifiable index)
- Heatmap

"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

from yellowbrick.utils import is_dataframe
from yellowbrick.utils import is_structured_array
from yellowbrick.features.base import DataVisualizer

# from yellowbrick.style.colors import resolve_colors



##########################################################################
## Feature Visualizer
##########################################################################


class MissingValuesBarVisualizer(DataVisualizer):
    """
    """

    def __init__(self,
                 ax=None,
                 x=None,
                 y=None,
                 features=None,
                 classes=None,
                 color=None,
                 colormap=None,
                 **kwargs):
        """
        """

        super(MissingValuesBarVisualizer, self).__init__(ax, features, classes, color,
                                                colormap, **kwargs)


    def fit(self, X, y=None, **kwargs):
        """
        TODO if y, then color code the missing values in the chart?


        """
        nrows, ncols = df.shape

        # Handle the feature names if they're None.
        if self.features_ is not None and is_dataframe(X):
            X = X[self.features_].as_matrix()

        # handle numpy named/ structured array
        elif self.features_ is not None and is_structured_array(X):
            X_selected = X[self.features_]
            X = X_selected.copy().view((np.float64, len(X_selected.dtype.names)))

        else:
            pass

        if self.features_ is None:
            self.features_ = range(nrows)

        if self.classes_ is None:
            # TODO: Is this the most efficient method?
            self.classes_ = [str(label) for label in np.unique(y)]

        nan_matrix = X.astype(float)
        self.nan_col_counts = [np.count_nonzero(np.isnan(col)) for col in nan_matrix.T]

        # Draw the instances
        self.draw(X, y, **kwargs)

        # Fit always returns self.
        return self

    def draw(self, X, y, **kwargs):
        """Called from the fit method, this method creates a scatter plot that
        draws each instance as a class or target colored point, whose location
        is determined by the feature data set.
        """

        width = 0.5  # the width of the bars

        self.ax.bar(ind - width/2, self.nan_col_counts, width,
                        color='black')

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
            'Missing Values by Column'
        )
        ind = np.arange(len(self.features_))  # the x locations for the groups
        # Remove the ticks from the graph
        self.ax.set_ylabel('Count')
        self.ax.set_xticks(ind)
        self.ax.set_xticklabels(self.features_, rotation='vertical')
        # Add the legend
        self.ax.legend(loc='best')
