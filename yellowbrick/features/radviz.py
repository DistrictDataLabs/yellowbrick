# yellowbrick.features.radviz
# Implements radviz for feature analysis.
#
# Author:   Benjamin Bengfort
# Created:  Fri Oct 07 13:18:00 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: radviz.py [0f4b236] benjamin@bengfort.com $

"""
Implements radviz for feature analysis.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.patches as patches

from yellowbrick.draw import manual_legend
from yellowbrick.utils import is_dataframe
from yellowbrick.utils import nan_warnings
from yellowbrick.features.base import DataVisualizer


##########################################################################
## Static RadViz Visualizer
##########################################################################


class RadialVisualizer(DataVisualizer):
    """
    RadViz is a multivariate data visualization algorithm that plots each
    axis uniformely around the circumference of a circle then plots points on
    the interior of the circle such that the point normalizes its values on
    the axes from the center to each arc.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    features : list, default: None
        a list of feature names to use
        The names of the features specified by the columns of the input dataset.
        This length of this list must match the number of columns in X, otherwise
        an exception will be raised on ``fit()``.

    classes : list, default: None
        a list of class names for the legend
        The class labels for each class in y, ordered by sorted class index. These
        names act as a label encoder for the legend, identifying integer classes
        or renaming string labels. If omitted, the class labels will be taken from
        the unique values in y.

        Note that the length of this list must match the number of unique values in
        y, otherwise an exception is raised. This parameter is only used in the
        discrete target type case and is ignored otherwise.

    colors : list or tuple, default: None
        optional list or tuple of colors to colorize lines
        A single color to plot all instances as or a list of colors to color each
        instance according to its class. If not enough colors per class are
        specified then the colors are treated as a cycle.

    colormap : string or cmap, default: None
        optional string or matplotlib cmap to colorize lines
        The colormap used to create the individual colors. If classes are
        specified the colormap is used to evenly space colors across each class.

    alpha : float, default: 1.0
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------

    >>> visualizer = RadViz()
    >>> visualizer.fit(X, y)
    >>> visualizer.transform(X)
    >>> visualizer.show()

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
    """

    def __init__(
        self,
        ax=None,
        features=None,
        classes=None,
        colors=None,
        colormap=None,
        alpha=1.0,
        **kwargs
    ):
        if "target_type" not in kwargs:
            kwargs["target_type"] = "discrete"
        super(RadialVisualizer, self).__init__(
            ax=ax,
            features=features,
            classes=classes,
            colors=colors,
            colormap=colormap,
            **kwargs
        )
        self.alpha = alpha

    @staticmethod
    def normalize(X):
        """
        MinMax normalization to fit a matrix in the space [0,1] by column.
        """
        a = X.min(axis=0)
        b = X.max(axis=0)
        return (X - a[np.newaxis, :]) / ((b - a)[np.newaxis, :])

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
        super(RadialVisualizer, self).fit(X, y)
        self.draw(X, y, **kwargs)
        return self

    def draw(self, X, y, **kwargs):
        """
        Called from the fit method, this method creates the radviz canvas and
        draws each instance as a class or target colored point, whose location
        is determined by the feature data set.
        """
        # Convert from dataframe
        if is_dataframe(X):
            X = X.values

        # Clean out nans and warn that the user they aren't plotted
        nan_warnings.warn_if_nans_exist(X)
        X, y = nan_warnings.filter_missing(X, y)

        # Get the shape of the data
        nrows, ncols = X.shape

        # Set the axes limits
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])

        # Create a data structure to hold scatter plot representations
        to_plot = {label: [[], []] for label in self.classes_}

        # Compute the arcs around the circumference for each feature axis
        # TODO: make this an independent function for override
        s = np.array(
            [
                (np.cos(t), np.sin(t))
                for t in [2.0 * np.pi * (i / float(ncols)) for i in range(ncols)]
            ]
        )

        # Compute the locations of the scatter plot for each class
        # Normalize the data first to plot along the 0, 1 axis
        for i, row in enumerate(self.normalize(X)):
            row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)
            xy = (s * row_).sum(axis=0) / row.sum()
            label = self._label_encoder[y[i]]

            to_plot[label][0].append(xy[0])
            to_plot[label][1].append(xy[1])

        # Add the scatter plots from the to_plot function
        # TODO: store these plots to add more instances to later
        # TODO: make this a separate function
        for label in self.classes_:
            color = self.get_colors([label])[0]
            self.ax.scatter(
                to_plot[label][0],
                to_plot[label][1],
                color=color,
                label=label,
                alpha=self.alpha,
                **kwargs
            )

        # Add the circular axis path
        # TODO: Make this a seperate function (along with labeling)
        self.ax.add_patch(
            patches.Circle(
                (0.0, 0.0),
                radius=1.0,
                facecolor="none",
                edgecolor="grey",
                linewidth=0.5,
            )
        )

        # Add the feature names
        for xy, name in zip(s, self.features_):
            # Add the patch indicating the location of the axis
            self.ax.add_patch(patches.Circle(xy, radius=0.025, facecolor="#777777"))

            # Add the feature names offset around the axis marker
            if xy[0] < 0.0 and xy[1] < 0.0:
                self.ax.text(
                    xy[0] - 0.025,
                    xy[1] - 0.025,
                    name,
                    ha="right",
                    va="top",
                    size="small",
                )
            elif xy[0] < 0.0 and xy[1] >= 0.0:
                self.ax.text(
                    xy[0] - 0.025,
                    xy[1] + 0.025,
                    name,
                    ha="right",
                    va="bottom",
                    size="small",
                )
            elif xy[0] >= 0.0 and xy[1] < 0.0:
                self.ax.text(
                    xy[0] + 0.025,
                    xy[1] - 0.025,
                    name,
                    ha="left",
                    va="top",
                    size="small",
                )
            elif xy[0] >= 0.0 and xy[1] >= 0.0:
                self.ax.text(
                    xy[0] + 0.025,
                    xy[1] + 0.025,
                    name,
                    ha="left",
                    va="bottom",
                    size="small",
                )

        self.ax.axis("equal")
        return self.ax

    def finalize(self, **kwargs):
        """
        Sets the title and adds a legend. Removes the ticks from the graph to
        make a cleaner visualization.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        Notes
        -----
        Generally this method is called from show and not directly by the user.
        """
        # Set the title
        self.set_title("RadViz for {} Features".format(len(self.features_)))

        # Remove the ticks from the graph
        self.ax.set_yticks([])
        self.ax.set_xticks([])

        # Add the legend
        colors = self.get_colors(self.classes_)
        manual_legend(self, self.classes_, colors, loc="best")


##########################################################################
## Quick Method
##########################################################################


def radviz(
    X,
    y=None,
    ax=None,
    features=None,
    classes=None,
    colors=None,
    colormap=None,
    alpha=1.0,
    show=True,
    **kwargs
):
    """
    Displays each feature as an axis around a circle surrounding a scatter
    plot whose points are each individual instance.

    This helper function is a quick wrapper to utilize the RadialVisualizer
    (Transformer) for one-off analysis.

    Parameters
    ----------

    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features

    y : ndarray or Series of length n, default:None
        An array or series of target or class values

    ax : matplotlib Axes, default: None
        The axes to plot the figure on.

    features : list of strings, default: None
        The names of the features or columns

    classes : list of strings, default: None
        The names of the classes in the target

    colors : list or tuple of colors, default: None
        Specify the colors for each individual class

    colormap : string or matplotlib cmap, default: None
        Sequential colormap for continuous target

    alpha : float, default: 1.0
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    show: bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot
        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply
        calls ``finalize()``
        
    kwargs : dict
        Keyword arguments passed to the visualizer base classes.

    Returns
    -------
    viz : RadViz
        Returns the fitted, finalized visualizer
    """
    # Instantiate the visualizer
    visualizer = RadialVisualizer(
        ax=ax,
        features=features,
        classes=classes,
        colors=colors,
        colormap=colormap,
        alpha=alpha,
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


# Alias for RadViz
RadViz = RadialVisualizer
