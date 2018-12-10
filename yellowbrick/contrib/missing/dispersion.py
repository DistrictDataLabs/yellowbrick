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
    >>> visualizer.poof()
    """

    def __init__(self, alpha=0.5, marker="|", classes=None, **kwargs):

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

        # properties set later
        self.nan_locs_ = []


    def get_nan_locs(self, X, y, **kwargs):
        """Gets the locations of nans in feature data and returns
        the coordinates in the matrix
        """
        nan_matrix = self.create_nan_matrix(X)

        if y is None:
            return np.argwhere(np.isnan(nan_matrix))
        else:
            self.nan_locs_ = []
            for target_value in np.unique(y):
                indices = np.argwhere(y == target_value)
                target_matrix = nan_matrix[indices.flatten()]
                nan_target_locs = np.argwhere(np.isnan(target_matrix))
                self.nan_locs_.append((target_value, nan_target_locs))

            return self.nan_locs_

    def draw(self, X, y, **kwargs):
        """Called from the fit method, this method creates a scatter plot that
        draws each instance as a class or target colored point, whose location
        is determined by the feature data set.

        If y is not None, then it draws a scatter plot where each class is in a
        different color.
        """
        nan_locs = self.get_nan_locs(X, y)
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
        # the x locations for the groups
        tick_locations = np.arange(len(self.features_))

        self.ax.set_xlabel('Position by index')
        self.ax.set_yticks(tick_locations)
        self.ax.set_yticklabels(self.get_feature_names())
        self.ax.legend(loc='upper left', prop={'size':5}, bbox_to_anchor=(1,1))



##########################################################################
## Quick Method
##########################################################################

def missing_dispersion(X, y=None, ax=None, classes=None, alpha=0.5, marker="|", **kwargs):
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
    visualizer.poof()

    # Return the axes object on the visualizer
    return visualizer.ax
