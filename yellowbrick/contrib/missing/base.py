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

    def fit(self, X, y=None, index_column=None, **kwargs):
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

        index_column : ndarray or Series of length n
            An array to be used as the x axis such as datetime

        kwargs : dict
            Pass generic arguments to the drawing method

        Returns
        -------
        self : instance
            Returns the instance of the transformer/visualizer
        """
        if is_dataframe(X):
            X_ = X.values
            if self.features_ is None:
                self.features_ = X.columns
        else:
            # ensure that X is an np array not a nested list
            X_ = np.array(X)

        return super(MissingDataVisualizer, self).fit(X_, y, index_column=index_column, **kwargs)


    def get_feature_names(self):
        if self.features_ is None:
            return ["Feature {}".format(str(n)) for n in np.arange(len(self.features_))]
        return self.features_

    def create_nan_matrix(self, X):
        """Given a numpy matrix X, creates a new matrix that contains nan values
        that can be of further use in missing values visualizers
        """
        # where matrix contains strings, handle them
        if np.issubdtype(X.dtype, np.string_) or np.issubdtype(X.dtype, np.unicode_):
            mask = np.where( X == '' )
            nan_matrix = np.zeros(X.shape)
            nan_matrix[mask] = np.nan

        else:
            nan_matrix = X.astype(np.float)

        return nan_matrix
