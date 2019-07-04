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
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from enum import Enum

from yellowbrick.base import Visualizer
from yellowbrick.utils import is_dataframe
from yellowbrick.style import resolve_colors
from yellowbrick.draw import manual_legend
from yellowbrick.exceptions import YellowbrickValueError, YellowbrickWarning,  NotFitted

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

class TargetType(Enum):
    AUTO = 'auto'
    SINGLE = 'single'
    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'


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

    target_type : str, default: "auto"
        Specify the type of target as either "discrete" (classes) or "continuous"
        (real numbers, usually for regression). If "auto", then it will
        attempt to determine the type by counting the number of unique values.

        If the target is discrete, the colors are returned as a dict with classes
        being the keys. If continuous the colors will be list having value of
        color for each point. In either case, if no target is specified, then
        color will be specified as blue.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Notes
    -----
        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.
    """

    def __init__(self, ax=None, features=None, classes=None, color=None,
                 colormap=None, target_type="auto", **kwargs):
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
        try:
            # Ensures that target is either Single, Discrete, Continuous or Auto
            self.target_type = TargetType(target_type)
        except ValueError:
            raise YellowbrickValueError("unknown target color type '{}'".format(target_type))

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

        self._determine_target_color_type(y)

        if self._target_color_type == TargetType.SINGLE:
            self._colors = 'b'

        # Compute classes and colors if target type is discrete
        elif self._target_color_type == TargetType.DISCRETE:
            # Store the classes for the legend if they're None.
            if self.classes_ is None:
                # TODO: Is this the most efficient method?
                self.classes_ = [str(label) for label in np.unique(y)]

            # Ensures that classes passed by user is equal to that in target
            if len(self.classes_)!=len(np.unique(y)):
                warnings.warn(("Number of unique target is not "
                              "equal to classes"), YellowbrickWarning)

            color_values = resolve_colors(n_colors=len(self.classes_),
                                          colormap=self.colormap, colors=self.color)
            self._colors = dict(zip(self.classes_, color_values))

        # Compute target range if colors are continuous
        elif self._target_color_type == TargetType.CONTINUOUS:
            y = np.asarray(y)
            self.range_ = (y.min(), y.max())

            self._colors = mpl.cm.get_cmap(self.colormap)

        else:
            raise YellowbrickValueError(
                    "unknown target color type '{}'".format(self._target_color_type))

        # Draw the instances
        self.draw(X, y, **kwargs)

        # Fit always returns self.
        return self

    def _determine_target_color_type(self, y):
        """
        Determines the target color type from the vector y as follows:

            - if y is None: only a single color is used
            - if target is auto: determine if y is continuous or discrete
            - otherwise specify supplied target type

        This property will be used to compute the colors for each point.
        """
        if y is None:
            self._target_color_type = TargetType.SINGLE
        elif self.target_type == TargetType.AUTO:
            # NOTE: See #73 for a generalization to use when implemented
            if len(np.unique(y)) < 10:
                self._target_color_type = TargetType.DISCRETE
            else:
                self._target_color_type = TargetType.CONTINUOUS
        else:
            self._target_color_type = self.target_type

        # Ensures that target is either SINGLE, DISCRETE or CONTINUOS and not AUTO
        if self._target_color_type == TargetType.AUTO:
            raise YellowbrickValueError((
                "could not determine target color type "
                "from target='{}' to '{}'"
            ).format(self.target_type, self._target_color_type))

class ProjectionVisualizer(DataVisualizer):
    
    def __init__(self, ax=None, features=None, classes=None, color=None,
             colormap=None, target_type="auto", proj_dim=2, **kwargs):
        super(ProjectionVisualizer, self).__init__(ax=ax, features=features, 
                                                     classes=classes, color=color,
                                                     colormap=colormap, 
                                                     target_type=target_type, **kwargs)
        
        if proj_dim not in [2, 3]:
            raise YellowbrickValueError("Projection dimensions must be either 2 or 3")
        self.proj_dim=proj_dim
        self._cax = None

    @property
    def cax(self):
        """
        The axes of the colorbar, right of the scatterplot.
        """
        if self._cax is None:
            raise AttributeError(
                "this visualizer does not have a histogram for the X axis"
            )
        return self._cax       
        
    def _layout_colorbar(self):
        """
        Creates the layout for colorbar. The colorbar is added to the right of the 
        scatterplot 
        """
        # Ensure matplotlib version compatibility
        if make_axes_locatable is None:
            raise YellowbrickValueError((
                "heatmap requires matplotlib 2.0.2 or greater "
                "please upgrade matplotlib or set heatmap=False on the visualizer"
            ))

        # Create the new axes for the colorbar
        divider = make_axes_locatable(self.ax)
        self._cax = divider.append_axes("right", size="5%", pad=0.3)
        
    def fit(self, X, y=None, **kwargs):
        
        self.fit_transform(X, y, **kwargs)
        return self
        
    def fit_transform(self, X, y=None, **kwargs):

        # Transformer will be  defined either in the subclasses or explicitly
        self.transformed_features_ = self.transformer.fit_transform(X)
        super(ProjectionVisualizer, self).fit(X, y, **kwargs)
        return self.transformed_features_    
        
    def transform(self, X, y=None, **kwargs):
        try:
            return self.transformer.transform(X)
        except AttributeError as e:
            raise AttributeError(str(e) + " try using fit_transform instead.")

    # Not sure about the function signature      
    def draw(self, y=None):
        X = self.transformed_features_
        scatter_kwargs = self._determine_scatter_kwargs(y);
        
        if self.proj_dim == 2:
            self._scatter = self.ax.scatter(X[:,0], X[:,1], **scatter_kwargs)

        if self.proj_dim == 3:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self._scatter = self.ax.scatter(X[:, 0], X[:, 1], X[:, 2], **scatter_kwargs)
        
        return self.ax

    def finalize(self):
        
        if self._target_color_type == TargetType.DISCRETE:
            # Add the legend
            manual_legend(self, self.classes_, list(self._colors.values()),
                          frameon=True)

        elif self._target_color_type == TargetType.CONTINUOUS:
            if(self.proj_dim==3):
                self.fig.colorbar(self._scatter, ax=self.ax)
            
            else:
                self._layout_colorbar()
                # Manually draw the colorbar.
                self.cbar = mpl.colorbar.ColorbarBase(self.cax, cmap=self._colors, 
                                                  norm=self._norm)

    def _determine_scatter_kwargs(self, y=None):
        scatter_kwargs = {"alpha": self.alpha}
        # Determine the colors
        if self._target_color_type == TargetType.SINGLE:
            scatter_kwargs["c"] = self._colors

        elif self._target_color_type == TargetType.DISCRETE:
            if y is None:
                raise YellowbrickValueError("y is required for discrete target")

            try:
                scatter_kwargs["c"] = [self._colors[self.classes_[yi]] for yi in y]
            except IndexError:
                raise YellowbrickValueError("Target needs to be label encoded.")


        elif self._target_color_type == TargetType.CONTINUOUS:
            if y is None:
                raise YellowbrickValueError("y is required for continuous target")

            scatter_kwargs["c"] = y
            scatter_kwargs["cmap"] = self._colors
            self._norm = mpl.colors.Normalize(vmin=self.range_[0], vmax=self.range_[1])

        else:
            # Technically this should never be raised
            raise NotFitted("could not determine target color type")
                # Draw the scatter plot with the associated colors and alpha
        return scatter_kwargs;