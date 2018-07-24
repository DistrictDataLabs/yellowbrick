# yellowbrick.contrib.missing.base
# Base Visualizer for missing values
#
# Author:  Nathan Danielsen <nathan.danielsen@gmail.com>
# Created: Fri Mar 29 5:17:36 2018 -0500
#
# Copyright (C) 2018 District Data Labs
# For license information, see LICENSE.txt
#
# ID: base.py [] nathan.danielsen@gmail.com.com $

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
        if is_dataframe(X):
            self.X = X.values
            if self.features_ is None:
                self.features_ = X.columns
        else:
            self.X = X

        self.y = y

        super(MissingDataVisualizer, self).fit(X, y, **kwargs)


    def get_feature_names(self):
        if self.features_ is None:
            return ["Feature {}".format(str(n)) for n in np.arange(len(self.features_))]
        return self.features_
