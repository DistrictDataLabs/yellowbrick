# yellowbrick.contrib.missing.base
# Base Visualizer for missing values
#
# Author:  Nathan Danielsen
# Created: Fri Mar 29 5:17:36 2018 -0500
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: base.py [1443e16] ndanielsen@users.noreply.github.com $

"""
Base classes for missing values visualizers.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from yellowbrick.features.base import DataVisualizer
from yellowbrick.utils import is_dataframe


##########################################################################
## Feature Visualizers
##########################################################################


class MissingDataVisualizer(DataVisualizer):
    """Base class for MissingDataVisualizers.
    """

    def fit(self, X, y=None, **kwargs):
        """
        The fit method is the primary drawing input for the
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
        -------
        self : instance
            Returns the instance of the transformer/visualizer
        """
        # Do not call super here - the data visualizer has been refactored
        # to provide increased functionality that is not yet compatible with
        # the current implementation. This mimicks the previous functionality.
        # TODO: Refactor MissingDataVisualizer to make use of new features.
        self.features_ = self.features

        if is_dataframe(X):
            self.X = X.values
            if self.features_ is None:
                self.features_ = X.columns
        else:
            self.X = X

        self.y = y

        self.draw(X, y, **kwargs)
        return self

    def get_feature_names(self):
        if self.features_ is None:
            return ["Feature {}".format(str(n)) for n in np.arange(len(self.features_))]
        return self.features_
