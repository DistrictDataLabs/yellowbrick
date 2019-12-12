# yellowbrick.features.projection
# Base class for all projection (decomposition) high dimensional data visualizers.
#
# Author:   Naresh Bachwani
# Created:  Wed Jul 17 08:59:33 2019 -0400
#
# Copyright (C) 2019, the scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: projection.py [21eb9d2] 43993586+naresh-bachwani@users.noreply.github.com $

"""
Base class for all projection (decomposition) high dimensional data visualizers.
"""

##########################################################################
## Imports
##########################################################################

import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mpl_toolkits.mplot3d  # noqa

from yellowbrick.draw import manual_legend
from yellowbrick.features.base import DataVisualizer, TargetType
from yellowbrick.exceptions import YellowbrickValueError, YellowbrickWarning, NotFitted


##########################################################################
## Projection Visualizers
##########################################################################


class ProjectionVisualizer(DataVisualizer):
    """
    The ProjectionVisualizer provides functionality for projecting a multi-dimensional
    dataset into either 2 or 3 components so they can be plotted as a scatter plot on
    2d or 3d axes. The visualizer acts as a transformer, and draws the transformed data
    on behalf of the user. Because it is a DataVisualizer, the ProjectionVisualizer
    can plot continuous scatter plots with a colormap or discrete scatter plots with
    a legend.

    This visualizer is a base class and is not intended to be uses directly.
    Subclasses should implement a ``transform()`` method that calls ``draw()`` using
    the transformed data and the optional target as input.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes.
        will be used (or generated if required).

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

    projection : int or string, default: 2
        The number of axes to project into, either 2d or 3d. To plot 3d plots
        with matplotlib, please ensure a 3d axes is passed to the visualizer,
        otherwise one will be created using the current figure.

    alpha : float, default: 0.75
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    colorbar : bool, default: True
        If the target_type is "continous" draw a colorbar to the right of the
        scatter plot. The colobar axes is accessible using the cax property.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.
    """

    def __init__(
        self,
        ax=None,
        features=None,
        classes=None,
        colors=None,
        colormap=None,
        target_type="auto",
        projection=2,
        alpha=0.75,
        colorbar=True,
        **kwargs
    ):

        super(ProjectionVisualizer, self).__init__(
            ax=ax,
            features=features,
            classes=classes,
            colors=colors,
            colormap=colormap,
            target_type=target_type,
            **kwargs
        )

        # Convert string to integer
        if isinstance(projection, str):
            if projection in {"2D", "2d"}:
                projection = 2
            if projection in {"3D", "3d"}:
                projection = 3
        if projection not in {2, 3}:
            raise YellowbrickValueError("Projection dimensions must be either 2 or 3")
        self.projection = projection

        if self.ax.name != "3d" and self.projection == 3:
            warnings.warn(
                "data projection to 3 dimensions requires a 3d axes to draw on.",
                YellowbrickWarning,
            )

        self.alpha = alpha
        self.colorbar = colorbar
        self._cax = None

    @property
    def cax(self):
        """
        The axes of the colorbar, right of the scatterplot.
        """
        if self._cax is None:
            raise AttributeError("This visualizer does not have an axes for colorbar")

        return self._cax

    @property
    def ax(self):
        """
        Overloads the axes property from base class. If no axes is specified then
        creates an axes for users. A 3d axes is created for 3 dimensional plots.
        """
        if not hasattr(self, "_ax") or self._ax is None:
            if self.projection == 3:
                fig = plt.gcf()
                self._ax = fig.add_subplot(111, projection="3d")
            else:
                self._ax = plt.gca()
        return self._ax

    @ax.setter
    def ax(self, ax):
        self._ax = ax

    def layout(self, divider=None):
        """
        Creates the layout for colorbar when target type is continuous.
        The colorbar is added to the right of the scatterplot.

        Subclasses can override this method to add other axes or layouts.

        Parameters
        ----------
        divider: AxesDivider
            An AxesDivider to be passed among all layout calls.
        """
        if (
            self._target_color_type == TargetType.CONTINUOUS
            and self.projection == 2
            and self.colorbar
            and self._cax is None
        ):
            # Ensure matplotlib version compatibility
            if make_axes_locatable is None:
                raise YellowbrickValueError(
                    (
                        "Colorbar requires matplotlib 2.0.2 or greater "
                        "please upgrade matplotlib"
                    )
                )

            # Create the new axes for the colorbar
            if divider is None:
                divider = make_axes_locatable(self.ax)

            self._cax = divider.append_axes("right", size="5%", pad=0.3)
            self._cax.set_yticks([])
            self._cax.set_xticks([])

    def fit_transform(self, X, y=None):
        """
        Fits the visualizer on the input data, and returns transformed X.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix or data frame of n instances with m features where m>2.

        y : array-like of shape (n,), optional
            A vector or series with target values for each instance in X. This
            vector is used to determine the color of the points in X.

        Returns
        -------
        Xprime : array-like of shape (n, 2)
            Returns the 2-dimensional embedding of the instances.
        """
        return self.fit(X, y).transform(X, y)

    def draw(self, Xp, y=None):
        """
        Draws the points described by Xp and colored by the points in y. Can be
        called multiple times before finalize to add more scatter plots to the
        axes, however ``fit()`` must be called before use.

        Parameters
        ----------
        Xp : array-like of shape (n, 2) or (n, 3)
            The matrix produced by the ``transform()`` method.

        y : array-like of shape (n,), optional
            The target, used to specify the colors of the points.

        Returns
        -------
        self.ax : matplotlib Axes object
            Returns the axes that the scatter plot was drawn on.
        """
        scatter_kwargs = self._determine_scatter_kwargs(y)

        # Draws the layout of the visualizer. It draws the axes for colorbars,
        # heatmap, etc.
        self.layout()

        if self.projection == 2:
            # Adds colorbar axis for continuous target type.
            self.ax.scatter(Xp[:, 0], Xp[:, 1], **scatter_kwargs)

        if self.projection == 3:
            self.ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], **scatter_kwargs)

        return self.ax

    def finalize(self):
        """
        Draws legends and colorbar for scatter plots.
        """
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        if self.projection == 3:
            self.ax.set_zticklabels([])

        if self._target_color_type == TargetType.DISCRETE:
            # Add the legend
            manual_legend(
                self, self.classes_, list(self._colors.values()), frameon=True
            )
        elif self._target_color_type == TargetType.CONTINUOUS:
            if self.colorbar:
                if self.projection == 3:
                    sm = plt.cm.ScalarMappable(cmap=self._colors, norm=self._norm)
                    # Avoid MPL TypeError: "You must first set_array for mappable"
                    sm.set_array([])
                    self.cbar = plt.colorbar(sm, ax=self.ax)

                else:
                    # Manually draw the colorbar.
                    self.cbar = mpl.colorbar.ColorbarBase(
                        self.cax, cmap=self._colors, norm=self._norm
                    )

    def _determine_scatter_kwargs(self, y=None):
        """
        Determines scatter argumnets to pass into ``plt.scatter()``. If y is
        discrete or single then determine colors. If continuous then determine
        colors and colormap.Also normalize to range

        Parameters
        ----------
        y : array-like of shape (n,), optional
            The target, used to specify the colors of the points for continuous
            target.
        """

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
        return scatter_kwargs
