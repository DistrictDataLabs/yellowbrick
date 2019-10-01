# yellowbrick.features.base
# Base classes for feature visualizers and feature selection tools.
#
# Author:   Rebecca Bilbro
# Author:   Benjamin Bengfort
# Created:  Fri Oct 07 13:41:24 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
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

from yellowbrick.base import Visualizer
from yellowbrick.utils import is_dataframe
from yellowbrick.style import resolve_colors
from yellowbrick.exceptions import NotFitted
from yellowbrick.utils.target import target_color_type, TargetType
from yellowbrick.exceptions import YellowbrickKeyError, YellowbrickValueError
from yellowbrick.style import palettes

from matplotlib.colors import Normalize
from sklearn.base import TransformerMixin


##########################################################################
## Feature Visualizers
##########################################################################


class FeatureVisualizer(Visualizer, TransformerMixin):
    """Base class for feature visualization.

    Feature engineering is primarily conceptualized as a transformation or
    extraction operation, e.g. some raw data is passed through a series of
    transformers and mappings to result in some final dataset which can be
    directly fitted to a model. Therefore feature visualizers are
    transformers and support the sklearn transformer interface by implementing
    a transform method.

    Subclasses of the FeatureVisualizer may call draw either from fit or from
    transform but must implement both so that they can be supported in pipeline
    objects. By default, the transform method of the visualizer is just a data
    pass through that ensures the visualizer can be placed into a feature
    extraction workflow.

    Parameters
    ----------
    ax : matplotlib.Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    fig : matplotlib Figure, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

    kwargs : dict
        Any additional keyword arguments to pass to the base Visualizer.
    """

    def __init__(self, ax=None, fig=None, **kwargs):
        super(FeatureVisualizer, self).__init__(ax=ax, fig=fig, **kwargs)

    def transform(self, X, y=None):
        """
        A pass-through to ensure that feature visualizers work in Pipelines.
        Subclasses may override this method to actually transform data or to
        call drawing methods. The transformer may also take an optional y
        argument if it is required for either transformation or drawing.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature dataset to be transformed.

        y : array-like, shape (n_samples,)
            Dependent target data associated with X, unused.

        Returns
        -------
        X : array-like, shape (n_samples, n_features)
            Returns the original dataset,  unmodified.
        """
        return X

    def fit_transform_show(self, X, y=None, **kwargs):
        """Fit, transform, then visualize data in one step.

        A helper method similar to ``fit_transform`` that allows you to fit,
        transform, and create a visualization in one simple step. Returns a
        transformed dataset similar to ``fit_transform``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature dataset for both training and transformation.

        y : array-like, shape (n_samples,)
            Dependent target dataset optionally used for training.

        kwargs : dict, optional
            Keyword arguments to pass to the ``show()`` method.

        Returns
        -------
        Xp : array-like, shape (m_samples, m_features)
            The transformed dataset X prime.
        """
        Xp = self.fit_transform(X, y)
        self.show(**kwargs)
        return Xp


class MultiFeatureVisualizer(FeatureVisualizer):
    """Direct visualization of a feature set.

    MultiFeatureVisualiers visualize several features at once, usually in order
    to compare the effectiveness of a subset of features to the superset. This
    type of visualizer provides base functionality for identifying the names of
    the features either directly from the data or from user supplied values. It
    also provides other functionality for feature selection, e.g. ensuring that
    a subset of features is used if specified by the user.

    Parameters
    ----------
    ax : matplotlib.Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    fig : matplotlib Figure, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

    features : list, default: None
        The names of the features specified by the columns of the input dataset.
        This length of this list must match the number of columns in X, otherwise
        an exception will be raised on ``fit()``.

    kwargs : dict, optional
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    features_ : ndarray, shape (n_features,)
        The names of the features discovered or used in the visualizer that
        can be used as an index to access or modify data in X. If a user passes
        feature names in, those features are used. Otherwise the columns of a
        DataFrame are used or just simply the indices of the data array.
    """

    def __init__(self, ax=None, fig=None, features=None, **kwargs):
        super(MultiFeatureVisualizer, self).__init__(ax=ax, fig=fig, **kwargs)

        # Data Parameters
        self.features = features

    def fit(self, X, y=None):
        """
        This method performs preliminary computations in order to set up the
        figure or perform other analyses. It can also call drawing methods in
        order to set up various non-instance related figure elements.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature dataset to be transformed.

        y : array-like, shape (n_samples,)
            Optional dependent target data associated with X.

        Returns
        -------
        self : MultiFeatureVisualizer
            Returns the visualizer/transformer for use in Pipelines and chaining.
        """
        n_columns = X.shape[1]

        if self.features is not None:
            # Use the user-specified features with some checking
            # TODO: allow the user specified features to filter the dataset
            if len(self.features) != n_columns:
                raise YellowbrickValueError(
                    (
                        "number of supplied feature names does not match the number "
                        "of columns in the training data."
                    )
                )

            self.features_ = np.array(self.features)

        else:
            # Attempt to determine the feature names from the input data
            if is_dataframe(X):
                self.features_ = np.array(X.columns)

            # Otherwise create numeric labels for each column.
            else:
                self.features_ = np.arange(0, n_columns)

        # Ensure super is called and fit is returned
        super(MultiFeatureVisualizer, self).fit(X, y)
        return self


##########################################################################
## Data Visualizers
##########################################################################


class DataVisualizer(MultiFeatureVisualizer):
    """Visualizations of instances in feature space.

    Data Visualizers plot instances in feature space (sometimes also referred
    to as data space). Feature space is a multi-dimensional space defined by
    the columns of the dataset ``X`` when passed to ``fit()`` and ``transform``.
    These instances and their features are directly plotted in a representation
    of the higher dimensional space.

    Instances can also be labeled by an target vectory, ``y``. The target is
    visualized in data space by color. For example a discrete target for
    classification problems will use categorical colors and a legend. A
    continuous target for regression problems will use sequential colors with
    a colormap.

    This class provides helper functionality related to target identification:
    whether or not the target is sequential or categorical, and mapping a
    color sequence or color set to the targets as appropriate. It also
    determines the scope of the target, e.g. the unique classes or the range
    of the dataset for use in specific visualizations.

    Parameters
    ----------
    ax : matplotlib.Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    fig : matplotlib Figure, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

    features : list, default: None
        The names of the features specified by the columns of the input dataset.
        This length of this list must match the number of columns in X, otherwise
        an exception will be raised on ``fit()``.

    classes : list, default: None
        The class labels for each class in y, ordered by sorted class index. These
        names act as a label encoder for the legend, identifying integer classes
        or renaming string labels. If omitted, the class labels will be taken from
        the unique values in y.

        Note that the length of this list must match the number of unique values in
        y, otherwise an exception is raised. This parameter is only used in the
        discrete target type case and is ignored otherwise.

    colors : list or tuple, default: None
        A single color to plot all instances as or a list of colors to color each
        instance according to its class in the discrete case or as an ordered
        colormap in the sequential case. If not enough colors per class are
        specified then the colors are treated as a cycle.

    colormap : string or cmap, default: None
        The colormap used to create the individual colors. In the discrete case
        it is used to compute the number of colors needed for each class and
        in the continuous case it is used to create a sequential color map based
        on the range of the target.

    target_type : str, default: "auto"
        Specify the type of target as either "discrete" (classes) or "continuous"
        (real numbers, usually for regression). If "auto", then it will
        attempt to determine the type by counting the number of unique values.

        If the target is discrete, the colors are returned as a dict with classes
        being the keys. If continuous the colors will be list having value of
        color for each point. In either case, if no target is specified, then
        color will be specified as the first color in the color cycle.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    features_ : ndarray, shape (n_features,)
        The names of the features discovered or used in the visualizer that
        can be used as an index to access or modify data in X. If a user passes
        feature names in, those features are used. Otherwise the columns of a
        DataFrame are used or just simply the indices of the data array.

    classes_ : ndarray, shape (n_classes,)
        The class labels that define the discrete values in the target. Only
        available if the target type is discrete. This is guaranteed to be
        strings even if the classes are a different type.

    range_ : (min y, max y)
        A tuple that describes the minimum and maximum values in the target.
        Only available if the target type is continuous.
    """

    def __init__(
        self,
        ax=None,
        fig=None,
        features=None,
        classes=None,
        colors=None,
        colormap=None,
        target_type="auto",
        **kwargs
    ):
        super(DataVisualizer, self).__init__(
            ax=ax, fig=fig, features=features, **kwargs
        )

        # Validate raises YellowbrickValueError if invalid
        TargetType.validate(target_type)

        # Data Parameters
        self.classes = classes
        self.target_type = target_type

        # Visual Parameters
        self.colors = colors
        self.colormap = colormap

        # Internal attributes
        self._colors = None
        self._target_color_type = None
        self._label_encoder = None

    def fit(self, X, y=None):
        """
        Fits the visualizer to the training data set by determining the
        target type, colors, classes, and range of the data to ensure that
        the visualizer can accurately portray the instances in data space.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        Returns
        -------
        self : DataVisualizer
            Returns the instance of the transformer/visualizer
        """
        # Compute the features from the data
        super(DataVisualizer, self).fit(X, y)

        # Determine the target color type
        self._determine_target_color_type(y)

        # Handle the single target color type
        if self._target_color_type == TargetType.SINGLE:
            # use the user supplied color or the first color in the color cycle
            self._colors = self.colors or "C0"

        # Compute classes and colors if target type is discrete
        elif self._target_color_type == TargetType.DISCRETE:
            # Unique labels are used both for validation and color mapping
            labels = np.unique(y)

            # Handle user supplied classes
            if self.classes is not None:
                self.classes_ = np.asarray([str(c) for c in self.classes])

                # Validate user supplied class labels
                if len(self.classes_) != len(labels):
                    raise YellowbrickValueError(
                        (
                            "number of specified classes does not match "
                            "number of unique values in target"
                        )
                    )

            # Get the string labels from the unique values in y
            else:
                self.classes_ = np.asarray([str(c) for c in labels])

            # Create a map of class labels to colors
            color_values = resolve_colors(
                n_colors=len(self.classes_), colormap=self.colormap, colors=self.colors
            )
            self._colors = dict(zip(self.classes_, color_values))
            self._label_encoder = dict(zip(labels, self.classes_))

        # Compute target range if colors are continuous
        elif self._target_color_type == TargetType.CONTINUOUS:
            y = np.asarray(y)
            self.range_ = (y.min(), y.max())
            if self.colormap is None:
                self.colormap = palettes.DEFAULT_SEQUENCE
            # TODO: allow for Yellowbrick palettes here as well
            self._colors = mpl.cm.get_cmap(self.colormap)

        # If this exception is raised a developer error has occurred because
        # unknown types should have errored when the type was determined.
        else:
            raise YellowbrickValueError(
                "unknown target color type '{}'".format(self._target_color_type)
            )

        # NOTE: cannot call draw in fit to support data transformers
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
            self._target_color_type = target_color_type(y)
        else:
            self._target_color_type = TargetType(self.target_type)

        # Ensures that target is either SINGLE, DISCRETE or CONTINUOUS before continuing
        if (
            self._target_color_type == TargetType.AUTO
            or self._target_color_type == TargetType.UNKNOWN
        ):
            raise YellowbrickValueError(
                (
                    "could not determine target color type " "from target='{}' to '{}'"
                ).format(self.target_type, self._target_color_type)
            )

    def get_target_color_type(self):
        """
        Returns the computed target color type if fitted or specified by the user.
        """
        if self._target_color_type is None:
            raise NotFitted("unknown target color type on unfitted visualizer")
        return self._target_color_type

    def get_colors(self, y):
        """
        Returns the color for the specified value(s) of y based on the learned
        colors property for any specified target type.

        Parameters
        ----------
        y : array-like
            The values of y to get the associated colors for.

        Returns
        -------
        colors : list
            Returns a list of colors for each value in y.
        """
        if self._colors is None:
            raise NotFitted("cannot determine colors on unfitted visualizer")

        if self._target_color_type == TargetType.SINGLE:
            return [self._colors] * len(y)

        if self._target_color_type == TargetType.DISCRETE:
            try:
                # Use the label encoder to get the class name (or use the value
                # if the label is not mapped in the encoder) then use the class
                # name to get the color from the color map.
                return [self._colors[self._label_encoder.get(yi, yi)] for yi in y]
            except KeyError:
                unknown = set(y) - set(self._label_encoder.keys())
                unknown = ", ".join(["'{}'".format(uk) for uk in unknown])
                raise YellowbrickKeyError(
                    "could not determine color for classes {}".format(unknown)
                )

        if self._target_color_type == TargetType.CONTINUOUS:
            # Normalize values into target range and compute colors from colormap
            norm = Normalize(*self.range_)
            return self._colors(norm(y))

        # This is a developer error, we should never get here!
        raise YellowbrickValueError(
            "unknown target color type '{}'".format(self._target_color_type)
        )
