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
from yellowbrick.features.base import DataVisualizer
from yellowbrick.exceptions import YellowbrickTypeError, NotFitted


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
        super(ScatterVisualizer, self).__init__(ax, features, classes, color,
                                                colormap, **kwargs)

        self.x = x
        self.y = y

        self.color = color
        self.colormap = colormap

        if self.x is not None and self.y is not None and self.features_ is not None:
            raise YellowbrickValueError(
                'Please specify x,y or features, not both.')

        if self.x is not None and self.y is not None and self.features_ is None:
            self.features_ = [self.x, self.y]



    def fit(self, X, y=None, **kwargs):
        """

        """

        # Handle the feature names if they're None.
        if self.features_ is not None and is_dataframe(X):
            X_two_cols = X[self.features_].as_matrix()

        # handle numpy named/ structured array
        elif self.features_ is not None and is_structured_array(X):
            X_selected = X[self.features_]
            X_two_cols = X_selected.copy().view((np.float64, len(X_selected.dtype.names)))

        # handle features that are numeric columns in ndarray matrix
        elif self.features_ is not None and has_ndarray_int_columns(self.features_, X):
            f_one, f_two = self.features_
            X_two_cols = X[:, [int(f_one), int(f_two)]]

        else:
            pass

        # Store the classes for the legend if they're None.
        if self.classes_ is None:
            # TODO: Is this the most efficient method?
            self.classes_ = [str(label) for label in np.unique(y)]

        # Draw the instances
        self.draw(X_two_cols, y, **kwargs)

        # Fit always returns self.
        return self

    def draw(self, X, y, **kwargs):
        """Called from the fit method, this method creates a scatter plot that
        draws each instance as a class or target colored point, whose location
        is determined by the feature data set.
        """
