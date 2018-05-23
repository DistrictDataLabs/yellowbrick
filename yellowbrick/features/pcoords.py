# yellowbrick.features.pcoords
# Implementations of parallel coordinates for feature analysis.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Mon Oct 03 21:46:06 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: pcoords.py [0f4b236] benjamin@bengfort.com $

"""
Implementations of parallel coordinates for multi-dimensional feature
analysis. There are a variety of parallel coordinates from Andrews Curves to
coordinates that optimize column order.
"""

##########################################################################
## Imports
##########################################################################

from copy import copy
from numpy import hstack, ones
from numpy.random import RandomState

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import Normalizer, StandardScaler

from yellowbrick.utils import is_dataframe, is_series
from yellowbrick.features.base import DataVisualizer
from yellowbrick.exceptions import YellowbrickTypeError, YellowbrickValueError
from yellowbrick.style.colors import resolve_colors


##########################################################################
## Quick Methods
##########################################################################

def parallel_coordinates(X, y, ax=None, features=None, classes=None,
                         normalize=None, sample=1.0, color=None, colormap=None,
                         vlines=True, vlines_kwds=None, **kwargs):
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

    color : list or tuple, default: None
        optional list or tuple of colors to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    colormap : string or cmap, default: None
        optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    vlines : boolean, default: True
        flag to determine vertical line display

    vlines_kwds : dict, default: None
        options to style or display the vertical lines, default: None

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the parallel coordinates were drawn on.
    """
    # Instantiate the visualizer
    visualizer = ParallelCoordinates(
        ax, features, classes, normalize, sample, color, colormap, vlines,
        vlines_kwds, **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y, **kwargs)
    visualizer.transform(X)

    # Return the axes object on the visualizer
    return visualizer.ax


##########################################################################
## Static Parallel Coordinates Visualizer
##########################################################################

class ParallelCoordinates(DataVisualizer):
    """
    Parallel coordinates displays each feature as a vertical axis spaced
    evenly along the horizontal, and each instance as a line drawn between
    each individual axis.

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

    color : list or tuple, default: None
        optional list or tuple of colors to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    colormap : string or cmap, default: None
        optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    vlines : boolean, default: True
        flag to determine vertical line display

    vlines_kwds : dict, default: None
        options to style or display the vertical lines, default: None

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    --------

    n_samples_ : int
        number of samples included in the visualization object

    Examples
    --------

    >>> visualizer = ParallelCoordinates()
    >>> visualizer.fit(X, y)
    >>> visualizer.transform(X)
    >>> visualizer.poof()

    Notes
    -----

    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """

    normalizers = {
        'minmax': MinMaxScaler(),
        'maxabs': MaxAbsScaler(),
        'standard': StandardScaler(),
        'l1': Normalizer('l1'),
        'l2': Normalizer('l2'),
    }

    def __init__(self, ax=None, features=None, classes=None, normalize=None,
                 sample=1.0, random_state=None, shuffle=False, color=None, colormap=None, vlines=True,
                 vlines_kwds=None, **kwargs):
        super(ParallelCoordinates, self).__init__(
            ax, features, classes, color, colormap, **kwargs
        )

        # Validate 'normalize' argument
        if normalize in self.normalizers or normalize is None:
            self.normalize = normalize
        else:
            raise YellowbrickValueError(
                "'{}' is an unrecognized normalization method"
                .format(normalize)
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
            raise YellowbrickTypeError(
                "`sample` parameter must be int or float"
            )
        self.sample = sample

        # Set sample parameters
        if isinstance(shuffle, bool):
            self.shuffle = shuffle
        else:
            raise YellowbrickTypeError(
                "`shuffle` parameter must be boolean"
            )
        if self.shuffle:
            if (random_state is None) or isinstance(random_state, int):
                self._rng = RandomState(random_state)
            elif isinstance(random_state, RandomState):
                self._rng = random_state
            else:
                raise YellowbrickTypeError(
                    "`random_state` parameter must be None, int, or np.random.RandomState"
                )
        else:
            self._rng = None

        # Visual Parameters
        self.show_vlines = vlines
        self.vlines_kwds = vlines_kwds or {
            'linewidth': 1, 'color': 'black'
        }

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

        # Convert from dataframe
        if is_dataframe(X):
            X = X.as_matrix()
        if is_series(y):
            y = y.as_matrix()

        # Subsample
        X, y = self._subsample(X, y)

        # Normalize
        if self.normalize is not None:
            X = self.normalizers[self.normalize].fit_transform(X)

        # the super method calls draw and returns self
        super(ParallelCoordinates, self).fit(X, y, **kwargs)

        # Fit always returns self.
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

        # Get shape of the sampled data
        nrows, ncols = X.shape

        # Create the xticks for each column
        # TODO: Allow the user to specify this feature
        increments = list(range(ncols))

        # Create the colors
        # TODO: Allow both colormap, listed colors, and palette definition
        # TODO: Make this an independent function or property for override!
        color_values = resolve_colors(
            n_colors=len(self.classes_), colormap=self.colormap, colors=self.color
        )
        colors = dict(zip(self.classes_, color_values))

        # TODO: Make this function compatible with DataFrames!
        # TODO: Make an independent function to allow addition of instances!

        # Prepare to flatten data within each class
        #   introduce separation between individual data points using None in x-values and arbitrary value (one) in
        #   y-values
        X_separated = hstack([X, ones((nrows, 1))])
        increments_separated = copy(increments)
        increments_separated.append(None)

        # Plot each class
        for label, color in sorted(colors.items()):
            y_as_str = y.astype('str')  # must be consistent with class conversion in DataVisualizer.fit()
            X_in_class = X_separated[y_as_str == label, :]
            increments_in_class = increments_separated * len(X_in_class)
            if len(X_in_class) > 0:
                self.ax.plot(increments_in_class, X_in_class.flatten(),
                             label=label, color=colors[label], alpha=0.25, linewidth=1, **kwargs)

        # Add the vertical lines
        # TODO: Make an independent function for override!
        if self.show_vlines:
            for idx in increments:
                self.ax.axvline(idx, **self.vlines_kwds)

        # Set the limits
        self.ax.set_xticks(increments)
        self.ax.set_xticklabels(self.features_)
        self.ax.set_xlim(increments[0], increments[-1])

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
            'Parallel Coordinates for {} Features'.format(len(self.features_))
        )

        # Set the legend and the grid
        self.ax.legend(loc='best')
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
