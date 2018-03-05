# yellowbrick.features.base
# Base classes for feature visualizers and feature selection tools.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Oct 07 13:41:24 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: base.py [2e898a6] benjamin@bengfort.com $

"""
Base classes for feature visualizers and feature selection tools.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from yellowbrick.base import Visualizer
from yellowbrick.utils import is_dataframe
from sklearn.base import TransformerMixin


##########################################################################
## Feature Visualizers
##########################################################################


class FeatureVisualizer(Visualizer, TransformerMixin):
    """
    Base class for feature visualization to investigate features
    individually or together.

    FeatureVisualizer is itself a transformer so that it can be used in
    a Scikit-Learn Pipeline to perform automatic visual analysis during build.

    Accepts as input a DataFrame or Numpy array.
    """

    def __init__(self, ax=None, **kwargs):
        super(FeatureVisualizer, self).__init__(ax=ax, **kwargs)

    def transform(self, X):
        """
        Primarily a pass-through to ensure that the feature visualizer will
        work in a pipeline setting. This method can also call drawing methods
        in order to ensure that the visualization is constructed.

        This method must return a numpy array with the same shape as X.
        """
        return X

    def fit_transform_poof(self, X, y=None, **kwargs):
        """
        Fit to data, transform it, then visualize it.

        Fits the visualizer to X and y with opetional parameters by passing in
        all of kwargs, then calls poof with the same kwargs. This method must
        return the result of the transform method.
        """
        Xp = self.fit_transform(X, y, **kwargs)
        self.poof(**kwargs)
        return Xp


class MultiFeatureVisualizer(FeatureVisualizer):
    """
    MultiFeatureVisualiers are a subclass of FeatureVisualizer that visualize
    several features at once. This class provides base functionality for
    getting the names of features for use in plot annotation.

    Parameters
    ----------

    ax: matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    features: list, default: None
        a list of feature names to use
        If a DataFrame is passed to fit and features is None, feature
        names are selected as the columns of the DataFrame.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    """

    def __init__(self, ax=None, features=None, **kwargs):
        super(MultiFeatureVisualizer, self).__init__(ax=ax, **kwargs)

        # Data Parameters
        self.features_ = features

    def fit(self, X, y=None, **fit_params):
        """
        This method performs preliminary computations in order to set up the
        figure or perform other analyses. It can also call drawing methods in
        order to set up various non-instance related figure elements.

        This method must return self.
        """

        # Handle the feature names if they're None.
        if self.features_ is None:

            # If X is a data frame, get the columns off it.
            if is_dataframe(X):
                self.features_ = np.array(X.columns)

            # Otherwise create numeric labels for each column.
            else:
                _, ncols = X.shape
                self.features_ = np.arange(0, ncols)

        return self

##########################################################################
## Data Visualizers
##########################################################################

class DataVisualizer(MultiFeatureVisualizer):
    """
    Data Visualizers are a subclass of Feature Visualizers which plot the
    instances in feature space (also called data space, hence the name of the
    visualizer). Feature space is a multi-dimensional space defined by the
    columns of the instance dependent vector input, X which is passed to
    ``fit()`` and ``transform()``. Instances can also be labeled by the target
    independent vector input, y which is only passed to ``fit()``. For that
    reason most Data Visualizers perform their drawing in ``fit()``.

    This class provides helper functionality related to target identification:
    whether or not the target is sequential or categorical, and mapping a
    color sequence or color set to the targets as appropriate. It also uses
    the fit method to call the drawing utilities.

    Parameters
    ----------

    ax: matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    features: list, default: None
        a list of feature names to use
        If a DataFrame is passed to fit and features is None, feature
        names are selected as the columns of the DataFrame.

    classes: list, default: None
        a list of class names for the legend
        If classes is None and a y value is passed to fit then the classes
        are selected from the target vector.

    color: list or tuple, default: None
        optional list or tuple of colors to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    colormap: string or cmap, default: None
        optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Notes
    -----
        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.
    """

    def __init__(self, ax=None, features=None, classes=None, color=None,
                 colormap=None, **kwargs):
        """
        Initialize the data visualization with many of the options required
        in order to make most visualizations work.
        """
        super(DataVisualizer, self).__init__(ax=ax, features=features, **kwargs)

        # Data Parameters
        self.classes_  = classes

        # Visual Parameters
        self.color = color
        self.colormap = colormap

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
        super(DataVisualizer, self).fit(X, y, **kwargs)

        # Store the classes for the legend if they're None.
        if self.classes_ is None:
            # TODO: Is this the most efficient method?
            self.classes_ = [str(label) for label in set(y)]

        # Draw the instances
        self.draw(X, y, **kwargs)

        # Fit always returns self.
        return self
