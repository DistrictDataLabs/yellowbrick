# yellowbrick.contrib.missing.bar
# Missing Values Bar Visualizer
#
# Author:  Nathan Danielsen
# Created: Fri Mar 29 5:17:36 2018 -0500
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: bar.py [1443e16] ndanielsen@users.noreply.github.com $

"""
Bar visualizer of missing values by column.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from yellowbrick.style.palettes import color_palette
from .base import MissingDataVisualizer


##########################################################################
## MissingValues Visualizer
##########################################################################


class MissingValuesBar(MissingDataVisualizer):
    """The MissingValues Bar visualizer creates a bar graph that lists the total
    count of missing values for each selected feature column.

    When y targets are supplied to fit, the output is a stacked bar chart where
    each color corresponds to the total NaNs for the feature in that column.

    Parameters
    ----------
    alpha : float, default: 0.5
        A value for bending elments with the background.

    marker : matplotlib marker, default: |
        The marker used for each element coordinate in the plot

    color : string, default: black
        The color for drawing the bar chart when the y targets are not passed to
        fit.

    colors : list, default: None
        The color palette for drawing a stack bar chart when the y targets
        are passed to fit.

    classes : list, default: None
        A list of class names for the legend.
        If classes is None and a y value is passed to fit then the classes
        are selected from the target vector.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    features_ : np.array
        The feature labels ranked according to their importance

    classes_ : np.array
        The class labels for each of the target values

    Examples
    --------

    >>> from yellowbrick.contrib.missing import MissingValuesBar
    >>> visualizer = MissingValuesBar()
    >>> visualizer.fit(X, y=y)
    >>> visualizer.show()
    """

    def __init__(self, width=0.5, color=None, colors=None, classes=None, **kwargs):

        if "target_type" not in kwargs:
            kwargs["target_type"] = "single"
        super(MissingValuesBar, self).__init__(**kwargs)
        self.width = width  # the width of the bars
        self.classes_ = classes
        self.ind = None

        # Convert to array if necessary to match estimator.classes_
        if self.classes_ is not None:
            self.classes_ = np.array(classes)

        # Set up classifier score visualization properties
        self.color = color
        if self.classes_ is not None:
            n_colors = len(self.classes_)
        else:
            n_colors = None

        self.colors = color_palette(kwargs.pop("colors", None), n_colors)

    def get_nan_col_counts(self, **kwargs):
        if np.issubdtype(self.X.dtype, np.floating) or np.issubdtype(
                self.X.dtype, np.integer
        ):
            nan_matrix = self.X.astype(np.float64)
        else:
            # where matrix contains strings, handle them
            mask = np.where((self.X == "") | (self.X == 'nan'))
            nan_matrix = np.zeros(self.X.shape)
            nan_matrix[mask] = np.nan

        if self.y is None:
            nan_col_counts = [np.count_nonzero(np.isnan(col)) for col in nan_matrix.T]
            return nan_col_counts

        else:
            # add in counting of np.nan per target y by column
            nan_counts = []
            for target_value in np.unique(self.y):

                indices = np.argwhere(self.y == target_value)
                target_matrix = nan_matrix[indices.flatten()]
                nan_col_counts = np.array(
                    [np.count_nonzero(np.isnan(col)) for col in target_matrix.T]
                )
                nan_counts.append((target_value, nan_col_counts))
            return nan_counts

    def draw(self, X, y, **kwargs):
        """Called from the fit method, this method generated a horizontal bar plot.

        If y is none, then draws a simple horizontal bar chart.
        If y is not none, then draws a stacked horizontal bar chart for each nan count per
        target values.
        """
        nan_col_counts = self.get_nan_col_counts()

        # the x locations for the groups
        self.ind = np.arange(len(self.features_))

        if y is None:
            self.ax.barh(
                self.ind - self.width / 2,
                nan_col_counts,
                self.width,
                color=self.color,
                label=None,
            )
        else:
            self.draw_stacked_bar(nan_col_counts)

    def draw_stacked_bar(self, nan_col_counts):
        """Draws a horizontal stacked bar chart with different colors
        for each count of nan values per label.
        """
        for index, nan_values in enumerate(nan_col_counts):
            label, nan_col_counts = nan_values

            if index == 0:
                # first draw should be at zero
                bottom_chart = np.zeros(nan_col_counts.shape)

            # if features passed in then, label as such
            if self.classes_ is not None:
                label = self.classes_[index]

            color = self.colors[index]

            self.ax.barh(
                self.ind - self.width / 2,
                nan_col_counts,
                self.width,
                color=color,
                label=label,
                left=bottom_chart,
            )

            # keep track of counts to build on stacked
            bottom_chart = nan_col_counts

    def finalize(self, **kwargs):
        """
        Sets a title and x-axis labels and adds a legend. Also ensures that the
        y tick values are correctly set to feature names.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        Notes
        -----
        Generally this method is called from show and not directly by the user.
        """
        # Set the title
        self.set_title("Count of Missing Values by Column")
        tick_locations = np.arange(
            len(self.features_)
        )  # the x locations for the groups
        self.ax.set_yticks(tick_locations)
        self.ax.set_yticklabels(self.get_feature_names())
        # Remove the ticks from the graph
        self.ax.set_xlabel("Count")

        self.ax.legend(loc="best")


##########################################################################
## Quick Method
##########################################################################


def missing_bar(X, y=None, ax=None, classes=None, width=0.5, color="black", **kwargs):
    """The MissingValues Bar visualizer creates a bar graph that lists the total
    count of missing values for each selected feature column.

    When y targets are supplied to fit, the output is a stacked bar chart where
    each color corresponds to the total NaNs for the feature in that column.

    Parameters
    ----------
    alpha : float, default: 0.5
        A value for bending elments with the background.

    marker : matplotlib marker, default: |
        The marker used for each element coordinate in the plot

    color : string, default: black
        The color for drawing the bar chart when the y targets are not passed to
        fit.

    colors : list, default: None
        The color pallette for drawing a stack bar chart when the y targets
        are passed to fit.

    classes : list, default: None
        A list of class names for the legend.
        If classes is None and a y value is passed to fit then the classes
        are selected from the target vector.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    features_ : np.array
        The feature labels ranked according to their importance

    classes_ : np.array
        The class labels for each of the target values

    Examples
    --------

    >>> from yellowbrick.contrib.missing import missing_bar
    >>> visualizer = missing_bar(X, y=y)
    """
    # Instantiate the visualizer
    visualizer = MissingValuesBar(
        ax=ax, classes=classes, width=width, color=color, **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y)
    visualizer.show()

    # Return the axes object on the visualizer
    return visualizer.ax
