# yellowbrick.contrib.missing.dispersion
# Missing Values Dispersion Visualizer
#
# Author:  Nathan Danielsen
# Created: Fri Mar 29 5:17:36 2018 -0500
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: dispersion.py [1443e16] ndanielsen@users.noreply.github.com $

"""
Dispersion visualizer for locations of missing values by column against index position.
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


class MissingValuesDispersion(MissingDataVisualizer):
    """
    The Missing Values Dispersion visualizer shows the locations of missing (nan)
    values in the feature dataset by the order of the index.

    When y targets are supplied to fit, the output dispersion plot is color
    coded according to the target y that the element refers to.

    Parameters
    ----------
    alpha : float, default: 0.5
        A value for bending elments with the background.

    marker : matplotlib marker, default: |
        The marker used for each element coordinate in the plot

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

    >>> from yellowbrick.contrib.missing import MissingValuesDispersion
    >>> visualizer = MissingValuesDispersion()
    >>> visualizer.fit(X, y=y)
    >>> visualizer.show()
    """

    def __init__(self, alpha=0.5, marker="|", classes=None, **kwargs):

        if "target_type" not in kwargs:
            kwargs["target_type"] = "single"
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

        self.colors = color_palette(kwargs.pop("colors", None), n_colors)

    def get_nan_locs(self, **kwargs):
        """Gets the locations of nans in feature data and returns
        the coordinates in the matrix
        """
        if np.issubdtype(self.X.dtype, np.string_) or np.issubdtype(
            self.X.dtype, np.unicode_
        ):
            mask = np.where(self.X == "")
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
            self.ax.scatter(x_, y_, alpha=self.alpha, marker=self.marker, label=None)
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
            self.ax.scatter(
                x_, y_, alpha=self.alpha, marker=self.marker, color=color, label=label
            )

    def finalize(self, **kwargs):
        """
        Sets the title and x-axis label and adds a legend. Also ensures that
        the y tick labels are set to the feature names.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        Notes
        -----
        Generally this method is called from show and not directly by the user.
        """
        # Set the title
        self.set_title("Dispersion of Missing Values by Feature")
        # the x locations for the groups
        tick_locations = np.arange(len(self.features_))

        self.ax.set_xlabel("Position by index")
        self.ax.set_yticks(tick_locations)
        self.ax.set_yticklabels(self.get_feature_names())
        self.ax.legend(loc="upper left", prop={"size": 5}, bbox_to_anchor=(1, 1))


##########################################################################
## Quick Method
##########################################################################


def missing_dispersion(
    X, y=None, ax=None, classes=None, alpha=0.5, marker="|", **kwargs
):
    """
    The Missing Values Dispersion visualizer shows the locations of missing (nan)
    values in the feature dataset by the order of the index.

    When y targets are supplied to fit, the output dispersion plot is color
    coded according to the target y that the element refers to.

    Parameters
    ----------
    alpha : float, default: 0.5
        A value for bending elments with the background.

    marker : matplotlib marker, default: |
        The marker used for each element coordinate in the plot

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

    >>> from yellowbrick.contrib.missing import missing_dispersion
    >>> visualizer = missing_dispersion(X, y=y)

    """
    # Instantiate the visualizer
    visualizer = MissingValuesDispersion(
        ax=ax, classes=classes, alpha=alpha, marker=marker, **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y)
    visualizer.show()

    # Return the axes object on the visualizer
    return visualizer.ax
