# yellowbrick.features.pcoords
# Implementations of parallel coordinates for feature analysis.
#
# Author:  Benjamin Bengfort
# Author:  @thekylesaurus
# Created: Mon Oct 03 21:46:06 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: pcoords.py [0f4b236] benjamin@bengfort.com $

"""
Implementation of parallel coordinates for multi-dimensional feature analysis.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from numpy.random import RandomState
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import Normalizer, StandardScaler

from yellowbrick.draw import manual_legend
from yellowbrick.features.base import DataVisualizer
from yellowbrick.utils import is_dataframe, is_series
from yellowbrick.exceptions import YellowbrickTypeError, YellowbrickValueError


##########################################################################
## Quick Methods
##########################################################################

def parallel_coordinates(
    X,
    y,
    ax=None,
    features=None,
    classes=None,
    normalize=None,
    sample=1.0,
    random_state=None,
    shuffle=False,
    colors=None,
    colormap=None,
    alpha=None,
    fast=False,
    vlines=True,
    vlines_kwds=None,
    show=True,
    **kwargs
):
    """Displays each feature as a vertical axis and each instance as a line.

    This helper function is a quick wrapper to utilize the ParallelCoordinates
    Visualizer (Transformer) for one-off analysis.

    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features

    y : ndarray or Series of length n
        An array or series of target or class values

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    features : list, default: None
        a list of feature names to use
        If a DataFrame is passed to fit and features is None, feature
        names are selected as the columns of the DataFrame.

    classes : list, default: None
        a list of class names for the legend
        If classes is None and a y value is passed to fit then the classes
        are selected from the target vector.

    normalize : string or None, default: None
        specifies which normalization method to use, if any
        Current supported options are 'minmax', 'maxabs', 'standard', 'l1',
        and 'l2'.

    sample : float or int, default: 1.0
        specifies how many examples to display from the data
        If int, specifies the maximum number of samples to display.
        If float, specifies a fraction between 0 and 1 to display.

    random_state : int, RandomState instance or None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random; only used if shuffle is True and sample < 1.0

    shuffle : boolean, default: True
        specifies whether sample is drawn randomly

    colors : list or tuple, default: None
        optional list or tuple of colors to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    colormap : string or cmap, default: None
        optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    alpha : float, default: None
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered lines more visible.
        If None, the alpha is set to 0.5 in "fast" mode and 0.25 otherwise.

    fast : bool, default: False
        Fast mode improves the performance of the drawing time of parallel
        coordinates but produces an image that does not show the overlap of
        instances in the same class. Fast mode should be used when drawing all
        instances is too burdensome and sampling is not an option.

    vlines : boolean, default: True
        flag to determine vertical line display

    vlines_kwds : dict, default: None
        options to style or display the vertical lines, default: None

    show : bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot
        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply
        calls ``finalize()``

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Returns
    -------
    viz : ParallelCoordinates
        Returns the fitted, finalized visualizer
    """
    # Instantiate the visualizer
    visualizer = ParallelCoordinates(
        ax=ax,
        features=features,
        classes=classes,
        normalize=normalize,
        sample=sample,
        random_state=random_state,
        shuffle=shuffle,
        colors=colors,
        colormap=colormap,
        alpha=alpha,
        fast=fast,
        vlines=vlines,
        vlines_kwds=vlines_kwds,
        **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y, **kwargs)
    visualizer.transform(X)

    if show:
        visualizer.show()
    else:
        visualizer.finalize()

    # Return the visualizer object
    return visualizer


##########################################################################
## Static Parallel Coordinates Visualizer
##########################################################################


class ParallelCoordinates(DataVisualizer):
    """
    Parallel coordinates displays each feature as a vertical axis spaced
    evenly along the horizontal, and each instance as a line drawn between
    each individual axis. This allows you to detect braids of similar instances
    and separability that suggests a good classification problem.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    features : list, default: None
        a list of feature names to use
        If a DataFrame is passed to fit and features is None, feature
        names are selected as the columns of the DataFrame.

    classes : list, default: None
        a list of class names for the legend
        The class labels for each class in y, ordered by sorted class index. These
        names act as a label encoder for the legend, identifying integer classes
        or renaming string labels. If omitted, the class labels will be taken from
        the unique values in y.

        Note that the length of this list must match the number of unique values in
        y, otherwise an exception is raised.

    normalize : string or None, default: None
        specifies which normalization method to use, if any
        Current supported options are 'minmax', 'maxabs', 'standard', 'l1',
        and 'l2'.

    sample : float or int, default: 1.0
        specifies how many examples to display from the data
        If int, specifies the maximum number of samples to display.
        If float, specifies a fraction between 0 and 1 to display.

    random_state : int, RandomState instance or None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random; only used if shuffle is True and sample < 1.0

    shuffle : boolean, default: True
        specifies whether sample is drawn randomly

    colors : list or tuple, default: None
        A single color to plot all instances as or a list of colors to color each
        instance according to its class. If not enough colors per class are
        specified then the colors are treated as a cycle.

    colormap : string or cmap, default: None
        The colormap used to create the individual colors. If classes are
        specified the colormap is used to evenly space colors across each class.

    alpha : float, default: None
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered lines more visible.
        If None, the alpha is set to 0.5 in "fast" mode and 0.25 otherwise.

    fast : bool, default: False
        Fast mode improves the performance of the drawing time of parallel
        coordinates but produces an image that does not show the overlap of
        instances in the same class. Fast mode should be used when drawing all
        instances is too burdensome and sampling is not an option.

    vlines : boolean, default: True
        flag to determine vertical line display

    vlines_kwds : dict, default: None
        options to style or display the vertical lines, default: None

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    n_samples_ : int
        number of samples included in the visualization object

    features_ : ndarray, shape (n_features,)
        The names of the features discovered or used in the visualizer that
        can be used as an index to access or modify data in X. If a user passes
        feature names in, those features are used. Otherwise the columns of a
        DataFrame are used or just simply the indices of the data array.

    classes_ : ndarray, shape (n_classes,)
        The class labels that define the discrete values in the target. Only
        available if the target type is discrete. This is guaranteed to be
        strings even if the classes are a different type.

    Examples
    --------

    >>> visualizer = ParallelCoordinates()
    >>> visualizer.fit(X, y)
    >>> visualizer.transform(X)
    >>> visualizer.show()
    """

    NORMALIZERS = {
        "minmax": MinMaxScaler(),
        "maxabs": MaxAbsScaler(),
        "standard": StandardScaler(),
        "l1": Normalizer("l1"),
        "l2": Normalizer("l2"),
    }

    def __init__(
        self,
        ax=None,
        features=None,
        classes=None,
        normalize=None,
        sample=1.0,
        random_state=None,
        shuffle=False,
        colors=None,
        colormap=None,
        alpha=None,
        fast=False,
        vlines=True,
        vlines_kwds=None,
        **kwargs
    ):
        if "target_type" not in kwargs:
            kwargs["target_type"] = "discrete"
        super(ParallelCoordinates, self).__init__(
            ax=ax,
            features=features,
            classes=classes,
            colors=colors,
            colormap=colormap,
            **kwargs
        )

        # Validate 'normalize' argument
        if normalize in self.NORMALIZERS or normalize is None:
            self.normalize = normalize
        else:
            raise YellowbrickValueError(
                "'{}' is an unrecognized normalization method".format(normalize)
            )

        # Validate 'sample' argument
        if isinstance(sample, int):
            if sample < 1:
                raise YellowbrickValueError(
                    "`sample` parameter of type `int` must be greater than 1"
                )
        elif isinstance(sample, float):
            if sample <= 0 or sample > 1:
                raise YellowbrickValueError(
                    "`sample` parameter of type `float` must be between 0 and 1"
                )
        else:
            raise YellowbrickTypeError("`sample` parameter must be int or float")
        self.sample = sample

        # Set sample parameters
        if isinstance(shuffle, bool):
            self.shuffle = shuffle
        else:
            raise YellowbrickTypeError("`shuffle` parameter must be boolean")
        if self.shuffle:
            if (random_state is None) or isinstance(random_state, int):
                self._rng = RandomState(random_state)
            elif isinstance(random_state, RandomState):
                self._rng = random_state
            else:
                raise YellowbrickTypeError(
                    "`random_state` must be None, int, or np.random.RandomState"
                )
        else:
            self._rng = None

        # Visual and drawing parameters
        self.fast = fast
        self.alpha = alpha
        self.show_vlines = vlines
        self.vlines_kwds = vlines_kwds or {"linewidth": 1, "color": "black"}

        # Internal properties
        self._increments = None
        self._colors = None

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
        # Determine the features, classes, and colors
        super(ParallelCoordinates, self).fit(X, y)

        # Convert from pandas data types
        if is_dataframe(X):
            X = X.values
        if is_series(y):
            y = y.values

        # Ticks for each feature specified
        self._increments = np.arange(len(self.features_))

        # Subsample instances
        X, y = self._subsample(X, y)

        # Normalize instances
        if self.normalize is not None:
            X = self.NORMALIZERS[self.normalize].fit_transform(X)

        self.draw(X, y, **kwargs)
        return self

    def draw(self, X, y, **kwargs):
        """
        Called from the fit method, this method creates the parallel
        coordinates canvas and draws each instance and vertical lines on it.

        Parameters
        ----------
        X : ndarray of shape n x m
            A matrix of n instances with m features

        y : ndarray of length n
            An array or series of target or class values

        kwargs : dict
            Pass generic arguments to the drawing method

        """
        if self.fast:
            return self.draw_classes(X, y, **kwargs)
        return self.draw_instances(X, y, **kwargs)

    def draw_instances(self, X, y, **kwargs):
        """
        Draw the instances colored by the target y such that each line is a
        single instance. This is the "slow" mode of drawing, since each
        instance has to be drawn individually. However, in so doing, the
        density of instances in braids is more apparent since lines have an
        independent alpha that is compounded in the figure.

        This is the default method of drawing.

        Parameters
        ----------
        X : ndarray of shape n x m
            A matrix of n instances with m features

        y : ndarray of length n
            An array or series of target or class values

        Notes
        -----
        This method can be used to draw additional instances onto the parallel
        coordinates before the figure is finalized.
        """
        # Get alpha from param or default
        alpha = self.alpha or 0.25

        for idx in range(len(X)):
            Xi = X[idx]
            yi = y[idx]
            color = self.get_colors([yi])[0]

            self.ax.plot(self._increments, Xi, color=color, alpha=alpha, **kwargs)

        return self.ax

    def draw_classes(self, X, y, **kwargs):
        """
        Draw the instances colored by the target y such that each line is a
        single class. This is the "fast" mode of drawing, since the number of
        lines drawn equals the number of classes, rather than the number of
        instances. However, this drawing method sacrifices inter-class density
        of points using the alpha parameter.

        Parameters
        ----------
        X : ndarray of shape n x m
            A matrix of n instances with m features

        y : ndarray of length n
            An array or series of target or class values
        """
        # Get alpha from param or default
        alpha = self.alpha or 0.5

        # Prepare to flatten data within each class:
        #   introduce separation between individual data points using None in
        #   x-values and arbitrary value (one) in y-values
        X_separated = np.hstack([X, np.ones((X.shape[0], 1))])
        increments_separated = self._increments.tolist()
        increments_separated.append(None)

        # Get the classes that exist in the dataset, y
        y_values = np.unique(y)

        # Plot each class as a single line plot
        for yi in y_values:
            color = self.get_colors([yi])[0]

            X_in_class = X_separated[y == yi, :]
            increments_in_class = increments_separated * len(X_in_class)
            if len(X_in_class) > 0:
                self.ax.plot(
                    increments_in_class,
                    X_in_class.flatten(),
                    linewidth=1,
                    color=color,
                    alpha=alpha,
                    **kwargs
                )

        return self.ax

    def finalize(self, **kwargs):
        """
        Performs the final rendering for the multi-axis visualization, including
        setting and rendering the vertical axes each instance is plotted on. Adds
        a title, a legend, and manages the grid.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        Notes
        -----
        Generally this method is called from show and not directly by the user.
        """
        # Set the title
        self.set_title(
            "Parallel Coordinates for {} Features".format(len(self.features_))
        )

        # Add the vertical lines
        # TODO: Make an independent function for override!
        if self.show_vlines:
            for idx in self._increments:
                self.ax.axvline(idx, **self.vlines_kwds)

        # Set the limits
        self.ax.set_xticks(self._increments)
        self.ax.set_xticklabels(self.features_)
        self.ax.set_xlim(self._increments[0], self._increments[-1])

        # Add the legend sorting classes by name
        labels = sorted(list(self._colors.keys()))
        colors = [self._colors[lbl] for lbl in labels]
        manual_legend(self, labels, colors, loc="best", frameon=True)

        # Add the grid view
        self.ax.grid()

    def _subsample(self, X, y):

        # Choose a subset of samples
        if isinstance(self.sample, int):
            n_samples = min([self.sample, len(X)])
        elif isinstance(self.sample, float):
            n_samples = int(len(X) * self.sample)

        if (n_samples < len(X)) and self.shuffle:
            indices = self._rng.choice(len(X), n_samples, replace=False)
        else:
            indices = slice(n_samples)
        X = X[indices, :]
        y = y[indices]

        self.n_samples_ = n_samples
        return X, y
