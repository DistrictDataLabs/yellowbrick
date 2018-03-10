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

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import Normalizer, StandardScaler

from yellowbrick.utils import is_dataframe
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
                 sample=1.0, color=None, colormap=None, vlines=True,
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

        # Visual Parameters
        self.show_vlines = vlines
        self.vlines_kwds = vlines_kwds or {
            'linewidth': 1, 'color': 'black'
        }

    def draw(self, X, y, **kwargs):
        """
        Called from the fit method, this method creates the parallel
        coordinates canvas and draws each instance and vertical lines on it.
        """
        # Convert from dataframe
        if is_dataframe(X):
            X = X.as_matrix()

        # Choose a subset of samples
        # TODO: allow selection of a random subset of samples instead of head

        if isinstance(self.sample, int):
            self.n_samples = min([self.sample, len(X)])
        elif isinstance(self.sample, float):
            self.n_samples = int(len(X) * self.sample)
        X = X[:self.n_samples, :]

        # Normalize
        if self.normalize is not None:
            X = self.normalizers[self.normalize].fit_transform(X)

        # Get the shape of the data
        nrows, ncols = X.shape

        # Create the xticks for each column
        # TODO: Allow the user to specify this feature
        x = list(range(ncols))

        # Create the colors
        # TODO: Allow both colormap, listed colors, and palette definition
        # TODO: Make this an independent function or property for override!
        color_values = resolve_colors(
            n_colors=len(self.classes_), colormap=self.colormap, colors=self.color
        )
        colors = dict(zip(self.classes_, color_values))

        # Track which labels are already in the legend
        used_legends = set([])

        # TODO: Make this function compatible with DataFrames!
        # TODO: Make an independent function to allow addition of instances!
        for idx, row in enumerate(X):
            # TODO: How to map classmap to labels?
            label = y[idx] # Get the label for the row
            label = self.classes_[label]

            if label not in used_legends:
                used_legends.add(label)
                self.ax.plot(x, row, color=colors[label], alpha=0.25, label=label, **kwargs)
            else:
                self.ax.plot(x, row, color=colors[label], alpha=0.25, **kwargs)

        # Add the vertical lines
        # TODO: Make an independent function for override!
        if self.show_vlines:
            for idx in x:
                self.ax.axvline(idx, **self.vlines_kwds)

        # Set the limits
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(self.features_)
        self.ax.set_xlim(x[0], x[-1])

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
