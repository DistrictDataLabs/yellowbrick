# yellowbrick.features.jointplot
# Implementation of joint plots for univariate and bivariate analysis.
#
# Author:   Prema Damodaran Roman
# Created:  Mon Apr 10 21:00:54 2017 -0400
#
# Copyright (C) 2017 The scikit-yb developers.
# For license information, see LICENSE.txt
#
# ID: jointplot.py [7f47800] pdamodaran@users.noreply.github.com $

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

try:
    # Only available in Matplotlib >= 2.0.2
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    make_axes_locatable = None

from .base import FeatureVisualizer

# from ..bestfit import draw_best_fit # TODO: return in #728
from ..utils.types import is_dataframe
from ..exceptions import YellowbrickValueError
from scipy.stats import pearsonr, spearmanr, kendalltau


# Default Colors
# TODO: should we reuse these colors?
FACECOLOR = "#FAFAFA"
HISTCOLOR = "#6897bb"


# Objects for export
__all__ = ["JointPlot", "JointPlotVisualizer", "joint_plot"]


##########################################################################
## Joint Plot Visualizer
##########################################################################


class JointPlot(FeatureVisualizer):
    """
    Joint plots are useful for machine learning on multi-dimensional data, allowing for
    the visualization of complex interactions between different data dimensions, their
    varying distributions, and even their relationships to the target variable for
    prediction.

    The Yellowbrick ``JointPlot`` can be used both for pairwise feature analysis and
    feature-to-target plots. For pairwise feature analysis, the ``columns`` argument can
    be used to specify the index of the two desired columns in ``X``. If ``y`` is also
    specified, the plot can be colored with a heatmap or by class. For feature-to-target
    plots, the user can provide either ``X`` and ``y`` as 1D vectors, or a ``columns``
    argument with an index to a single feature in ``X`` to be plotted against ``y``.

    Histograms can be included by setting the ``hist`` argument to ``True`` for a
    frequency distribution, or to ``"density"`` for a probability density function. Note
    that histograms requires matplotlib 2.0.2 or greater.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes will be
        used (or generated if required). This is considered the base axes where the
        the primary joint plot is drawn. It will be shifted and two additional axes
        added above (xhax) and to the right (yhax) if hist=True.

    columns : int, str, [int, int], [str, str], default: None
        Determines what data is plotted in the joint plot and acts as a selection index
        into the data passed to ``fit(X, y)``. This data therefore must be indexable by
        the column type (e.g. an int for a numpy array or a string for a DataFrame).

        If None is specified then either both X and y must be 1D vectors and they will
        be plotted against each other or X must be a 2D array with only 2 columns. If a
        single index is specified then the data is indexed as ``X[columns]`` and plotted
        jointly with the target variable, y. If two indices are specified then they are
        both selected from X, additionally in this case, if y is specified, then it is
        used to plot the color of points.

        Note that these names are also used as the x and y axes labels if they aren't
        specified in the joint_kws argument.

    correlation : str, default: 'pearson'
        The algorithm used to compute the relationship between the variables in the
        joint plot, one of: 'pearson', 'covariance', 'spearman', 'kendalltau'.

    kind : str in {'scatter', 'hex'}, default: 'scatter'
        The type of plot to render in the joint axes. Note that when kind='hex' the
        target cannot be plotted by color.

    hist : {True, False, None, 'density', 'frequency'}, default: True
        Draw histograms showing the distribution of the variables plotted jointly.
        If set to 'density', the probability density function will be plotted.
        If set to True or 'frequency' then the frequency will be plotted.
        Requires Matplotlib >= 2.0.2.

    alpha : float, default: 0.65
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    {joint, hist}_kws : dict, default: None
        Additional keyword arguments for the plot components.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    corr_ : float
        The correlation or relationship of the data in the joint plot, specified by the
        correlation algorithm.

    Examples
    --------

    >>> viz = JointPlot(columns=["temp", "humidity"])
    >>> viz.fit(X, y)
    >>> viz.show()
    """

    # TODO: should we couple more closely with Rank2D?
    correlation_methods = {
        "pearson": lambda x, y: pearsonr(x, y)[0],
        "spearman": lambda x, y: spearmanr(x, y)[0],
        "covariance": lambda x, y: np.cov(x, y)[0, 1],
        "kendalltau": lambda x, y: kendalltau(x, y)[0],
    }

    def __init__(
        self,
        ax=None,
        columns=None,
        correlation="pearson",
        kind="scatter",
        hist=True,
        alpha=0.65,
        joint_kws=None,
        hist_kws=None,
        **kwargs
    ):
        # Initialize the visualizer
        super(JointPlot, self).__init__(ax=ax, **kwargs)
        self._xhax, self._yhax = None, None

        # Set and validate the columns
        self.columns = columns
        if self.columns is not None and not isinstance(self.columns, (int, str)):
            self.columns = tuple(self.columns)
            if len(self.columns) > 2:
                raise YellowbrickValueError(
                    (
                        "'{}' contains too many indices or is invalid for joint plot - "
                        "specify either a single int or str index or two columns as a list"
                    ).format(columns)
                )

        # Seet and validate the correlation
        self.correlation = correlation
        if self.correlation not in self.correlation_methods:
            raise YellowbrickValueError(
                "'{}' is an invalid correlation method, use one of {}".format(
                    self.correlation, ", ".join(self.correlation_methods.keys())
                )
            )

        # Set and validate the kind of plot
        self.kind = kind
        if self.kind not in {"scatter", "hex", "hexbin"}:
            raise YellowbrickValueError(
                ("'{}' is invalid joint plot kind, use 'scatter' or 'hex'").format(
                    self.kind
                )
            )

        # Set and validate the histogram if specified
        self.hist = hist
        if self.hist not in {True, "density", "frequency", None, False}:
            raise YellowbrickValueError(
                (
                    "'{}' is an invalid argument for hist, use None, True, "
                    "False, 'density', or 'frequency'"
                ).format(hist)
            )

        # If hist is True, test the version availability
        if self.hist in {True, "density", "frequency"}:
            self._layout()

        # Set the additional visual parameters
        self.alpha = alpha
        self.joint_kws = joint_kws
        self.hist_kws = hist_kws

    @property
    def xhax(self):
        """
        The axes of the histogram for the top of the JointPlot (X-axis)
        """
        if self._xhax is None:
            raise AttributeError(
                "this visualizer does not have a histogram for the X axis"
            )
        return self._xhax

    @property
    def yhax(self):
        """
        The axes of the histogram for the right of the JointPlot (Y-axis)
        """
        if self._yhax is None:
            raise AttributeError(
                "this visualizer does not have a histogram for the Y axis"
            )
        return self._yhax

    def _layout(self):
        """
        Creates the grid layout for the joint plot, adding new axes for the histograms
        if necessary and modifying the aspect ratio. Does not modify the axes or the
        layout if self.hist is False or None.
        """
        # Ensure the axes are created if not hist, then return.
        if not self.hist:
            self.ax
            return

        # Ensure matplotlib version compatibility
        if make_axes_locatable is None:
            raise YellowbrickValueError(
                (
                    "joint plot histograms requires matplotlib 2.0.2 or greater "
                    "please upgrade matplotlib or set hist=False on the visualizer"
                )
            )

        # Create the new axes for the histograms
        divider = make_axes_locatable(self.ax)
        self._xhax = divider.append_axes("top", size=1, pad=0.1, sharex=self.ax)
        self._yhax = divider.append_axes("right", size=1, pad=0.1, sharey=self.ax)

        # Modify the display of the axes
        self._xhax.xaxis.tick_top()
        self._yhax.yaxis.tick_right()
        self._xhax.grid(False, axis="y")
        self._yhax.grid(False, axis="x")

    def fit(self, X, y=None):
        """
        Fits the JointPlot, creating a correlative visualization between the columns
        specified during initialization and the data and target passed into fit:

            - If self.columns is None then X and y must both be specified as 1D arrays
              or X must be a 2D array with only 2 columns.
            - If self.columns is a single int or str, that column is selected to be
              visualized against the target y.
            - If self.columns is two ints or strs, those columns are visualized against
              each other. If y is specified then it is used to color the points.

        This is the main entry point into the joint plot visualization.

        Parameters
        ----------
        X : array-like
            An array-like object of either 1 or 2 dimensions depending on self.columns.
            Usually this is a 2D table with shape (n, m)

        y : array-like, default: None
            An vector or 1D array that has the same length as X. May be used to either
            directly plot data or to color data points.
        """
        # Convert python objects to numpy arrays
        if isinstance(X, (list, tuple)):
            X = np.array(X)

        if y is not None and isinstance(y, (list, tuple)):
            y = np.array(y)

        # Case where no columns are specified
        if self.columns is None:
            if (y is None and (X.ndim != 2 or X.shape[1] != 2)) or (
                y is not None and (X.ndim != 1 or y.ndim != 1)
            ):
                raise YellowbrickValueError(
                    (
                        "when self.columns is None specify either X and y as 1D arrays "
                        "or X as a matrix with 2 columns"
                    )
                )

            if y is None:
                # Draw the first column as x and the second column as y
                self.draw(X[:, 0], X[:, 1], xlabel="0", ylabel="1")
                return self

            # Draw x against y
            self.draw(X, y, xlabel="x", ylabel="y")
            return self

        # Case where a single string or int index is specified
        if isinstance(self.columns, (int, str)):
            if y is None:
                raise YellowbrickValueError(
                    "when self.columns is a single index, y must be specified"
                )

            # fetch the index from X -- raising index error if not possible
            x = self._index_into(self.columns, X)
            self.draw(x, y, xlabel=str(self.columns), ylabel="target")
            return self

        # Case where there is a double index for both columns
        columns = tuple(self.columns)
        if len(columns) != 2:
            raise YellowbrickValueError(
                ("'{}' contains too many indices or is invalid for joint plot").format(
                    columns
                )
            )

        # TODO: color the points based on the target if it is given
        x = self._index_into(columns[0], X)
        y = self._index_into(columns[1], X)
        self.draw(x, y, xlabel=str(columns[0]), ylabel=str(columns[1]))
        return self

    def draw(self, x, y, xlabel=None, ylabel=None):
        """
        Draw the joint plot for the data in x and y.

        Parameters
        ----------
        x, y : 1D array-like
            The data to plot for the x axis and the y axis

        xlabel, ylabel : str
            The labels for the x and y axes.
        """
        # This is a little weird to be here, but it is the best place to perform
        # this computation given how fit calls draw and returns.
        self.corr_ = self.correlation_methods[self.correlation](x, y)

        # First draw the joint plot
        joint_kws = self.joint_kws or {}
        joint_kws.setdefault("alpha", self.alpha)
        joint_kws.setdefault("label", "{}={:0.3f}".format(self.correlation, self.corr_))

        # Draw scatter joint plot
        if self.kind == "scatter":
            self.ax.scatter(x, y, **joint_kws)

            # TODO: Draw best fit line (or should this be kind='reg'?)

        # Draw hexbin joint plot
        elif self.kind in ("hex", "hexbin"):
            joint_kws.setdefault("mincnt", 1)
            joint_kws.setdefault("gridsize", 50)
            joint_kws.setdefault("cmap", "Blues")
            self.ax.hexbin(x, y, **joint_kws)

        # Something bad happened
        else:
            raise ValueError("unknown joint plot kind '{}'".format(self.kind))

        # Set the X and Y axis labels on the plot
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        # If we're not going to draw histograms, stop here
        if not self.hist:
            # Ensure the current axes is always the main joint plot axes
            plt.sca(self.ax)
            return self.ax

        # Draw the histograms
        hist_kws = self.hist_kws or {}
        hist_kws.setdefault("bins", 50)
        if self.hist == "density":
            hist_kws.setdefault("density", True)

        self.xhax.hist(x, **hist_kws)
        self.yhax.hist(y, orientation="horizontal", **hist_kws)

        # Ensure the current axes is always the main joint plot axes
        plt.sca(self.ax)
        return self.ax

    def finalize(self, **kwargs):
        """
        Finalize executes any remaining image modifications making it ready to show.
        """
        # Set the aspect ratio to make the visualization square
        # TODO: still unable to make plot square using make_axes_locatable
        # x0,x1 = self.ax.get_xlim()
        # y0,y1 = self.ax.get_ylim()
        # self.ax.set_aspect(abs(x1-x0)/abs(y1-y0))

        # Add the title to the plot if the user has set one.
        self.set_title("")

        # TODO: use manual legend so legend works with both scatter and hexbin
        # Set the legend with full opacity patches using manual legend.
        # Or Add the colorbar if this is a continuous plot.
        if self.kind == "scatter":
            self.ax.legend(loc="best", frameon=True)

        # Finalize the histograms
        if self.hist:
            plt.setp(self.xhax.get_xticklabels(), visible=False)
            plt.setp(self.yhax.get_yticklabels(), visible=False)
            plt.sca(self.ax)

        # Call tight layout to maximize readability
        self.fig.tight_layout()

    def _index_into(self, idx, data):
        """
        Attempts to get the column from the data using the specified index, raises an
        exception if this is not possible from this point in the stack.
        """
        try:
            if is_dataframe(data):
                # Assume column indexing
                return data[idx]
            # Otherwise assume numpy array-like indexing
            return data[:, idx]
        except Exception as e:
            raise IndexError(
                "could not index column '{}' into type {}: {}".format(
                    self.columns, data.__class__.__name__, e
                )
            )


# Alias for JointPlot
JointPlotVisualizer = JointPlot


##########################################################################
## Quick Method for JointPlot visualizations
##########################################################################


def joint_plot(
        X,
        y,
        ax=None,
        columns=None,
        correlation="pearson",
        kind="scatter",
        hist=True,
        alpha=0.65,
        joint_kws=None,
        hist_kws=None,
        show=True,
        **kwargs
):
    """
    Joint plots are useful for machine learning on multi-dimensional data, allowing for
    the visualization of complex interactions between different data dimensions, their
    varying distributions, and even their relationships to the target variable for
    prediction.

    The Yellowbrick ``JointPlot`` can be used both for pairwise feature analysis and
    feature-to-target plots. For pairwise feature analysis, the ``columns`` argument can
    be used to specify the index of the two desired columns in ``X``. If ``y`` is also
    specified, the plot can be colored with a heatmap or by class. For feature-to-target
    plots, the user can provide either ``X`` and ``y`` as 1D vectors, or a ``columns``
    argument with an index to a single feature in ``X`` to be plotted against ``y``.

    Histograms can be included by setting the ``hist`` argument to ``True`` for a
    frequency distribution, or to ``"density"`` for a probability density function. Note
    that histograms requires matplotlib 2.0.2 or greater.

    Parameters
    ----------
    X : array-like
        An array-like object of either 1 or 2 dimensions depending on self.columns.
        Usually this is a 2D table with shape (n, m)

    y : array-like, default: None
        An vector or 1D array that has the same length as X. May be used to either
        directly plot data or to color data points.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes will be
        used (or generated if required). This is considered the base axes where the
        the primary joint plot is drawn. It will be shifted and two additional axes
        added above (xhax) and to the right (yhax) if hist=True.

    columns : int, str, [int, int], [str, str], default: None
        Determines what data is plotted in the joint plot and acts as a selection index
        into the data passed to ``fit(X, y)``. This data therefore must be indexable by
        the column type (e.g. an int for a numpy array or a string for a DataFrame).

        If None is specified then either both X and y must be 1D vectors and they will
        be plotted against each other or X must be a 2D array with only 2 columns. If a
        single index is specified then the data is indexed as ``X[columns]`` and plotted
        jointly with the target variable, y. If two indices are specified then they are
        both selected from X, additionally in this case, if y is specified, then it is
        used to plot the color of points.

        Note that these names are also used as the x and y axes labels if they aren't
        specified in the joint_kws argument.

    correlation : str, default: 'pearson'
        The algorithm used to compute the relationship between the variables in the
        joint plot, one of: 'pearson', 'covariance', 'spearman', 'kendalltau'.

    kind : str in {'scatter', 'hex'}, default: 'scatter'
        The type of plot to render in the joint axes. Note that when kind='hex' the
        target cannot be plotted by color.

    hist : {True, False, None, 'density', 'frequency'}, default: True
        Draw histograms showing the distribution of the variables plotted jointly.
        If set to 'density', the probability density function will be plotted.
        If set to True or 'frequency' then the frequency will be plotted.
        Requires Matplotlib >= 2.0.2.

    alpha : float, default: 0.65
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    {joint, hist}_kws : dict, default: None
        Additional keyword arguments for the plot components.

    show : bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot
        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply
        calls ``finalize()``

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    corr_ : float
        The correlation or relationship of the data in the joint plot, specified by the
        correlation algorithm.
    """
    visualizer = JointPlot(
        ax=ax,
        columns=columns,
        correlation=correlation,
        kind=kind,
        hist=hist,
        alpha=alpha,
        joint_kws=joint_kws,
        hist_kws=hist_kws,
        **kwargs
    )

    # Fit and show the visualizer
    visualizer.fit(X, y)
    if show:
        visualizer.show()
    else:
        visualizer.finalize()
    return visualizer
