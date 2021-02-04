# yellowbrick.features.rankd
# Implements 1D (histograms) and 2D (joint plot) feature rankings.
#
# Author:   Benjamin Bengfort
# Created:  Fri Oct 07 15:14:01 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: rankd.py [ee754dc] benjamin@bengfort.com $

"""
Implements 1D (histograms) and 2D (joint plot) feature rankings.
"""

##########################################################################
## Imports
##########################################################################

import warnings
import numpy as np
import matplotlib as mpl

from scipy.stats import shapiro
from scipy.stats import spearmanr
from scipy.stats import kendalltau as sp_kendalltau

from yellowbrick.utils import is_dataframe
from yellowbrick.features.base import MultiFeatureVisualizer
from yellowbrick.exceptions import YellowbrickValueError, YellowbrickWarning


__all__ = ["rank1d", "rank2d", "Rank1D", "Rank2D"]


##########################################################################
## Metrics
##########################################################################


def kendalltau(X):
    """
    Accepts a matrix X and returns a correlation matrix so that each column
    is the variable and each row is the observations.

    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features

    """
    corrs = np.zeros((X.shape[1], X.shape[1]))
    for idx, cola in enumerate(X.T):
        for jdx, colb in enumerate(X.T):
            corrs[idx, jdx] = sp_kendalltau(cola, colb)[0]
    return corrs


##########################################################################
## Base Feature Visualizer
##########################################################################


class RankDBase(MultiFeatureVisualizer):
    """
    Base visualizer for Rank1D and Rank2D

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    fig : matplotlib Figure, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

    algorithm : string
        The ranking algorithm to use; options and defaults vary by subclass

    features : list
        A list of feature names to use.
        If a DataFrame is passed to fit and features is None, feature
        names are selected as the columns of the DataFrame.

    show_feature_names : boolean, default: True
        If True, the feature names are used to label the axis ticks in the
        plot.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    ranks_ : ndarray
        An n-dimensional, symmetric array of rank scores, where n is the
        number of features. E.g. for 1D ranking, it is (n,), for a
        2D ranking it is (n,n) and so forth.

    Examples
    --------

    >>> visualizer = Rank2D()
    >>> visualizer.fit(X, y)
    >>> visualizer.transform(X)
    >>> visualizer.show()

    Notes
    -----
    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """

    ranking_methods = {}

    def __init__(
        self,
        ax=None,
        fig=None,
        algorithm=None,
        features=None,
        show_feature_names=True,
        **kwargs
    ):
        """
        Initialize the class with the options required to rank and
        order features as well as visualize the result.
        """
        super(RankDBase, self).__init__(ax=ax, fig=fig, features=features, **kwargs)

        # Data Parameters
        self.ranking_ = algorithm

        # Display parameters
        self.show_feature_names_ = show_feature_names

    def transform(self, X, **kwargs):
        """
        The transform method is the primary drawing hook for ranking classes.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        kwargs : dict
            Pass generic arguments to the drawing method

        Returns
        -------
        X : ndarray
            Typically a transformed matrix, X' is returned. However, this
            method performs no transformation on the original data, instead
            simply ranking the features that are in the input data and returns
            the original data, unmodified.
        """
        self.ranks_ = self.rank(X)
        self.draw(**kwargs)

        # Return the X matrix, unchanged
        return X

    def rank(self, X, algorithm=None):
        """
        Returns the feature ranking.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        algorithm : str or None
            The ranking mechanism to use, or None for the default

        Returns
        -------
        ranks : ndarray
            An n-dimensional, symmetric array of rank scores, where n is the
            number of features. E.g. for 1D ranking, it is (n,), for a
            2D ranking it is (n,n) and so forth.
        """
        algorithm = algorithm or self.ranking_
        algorithm = algorithm.lower()

        if algorithm not in self.ranking_methods:
            raise YellowbrickValueError(
                "'{}' is unrecognized ranking method".format(algorithm)
            )

        # Extract matrix from dataframe if necessary
        if is_dataframe(X):
            X = X.values

        return self.ranking_methods[algorithm](X)

    def finalize(self, **kwargs):
        """
        Sets a title on the RankD plot.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        Notes
        -----
        Generally this method is called from show and not directly by the user.
        """
        # There is a known bug in matplotlib 3.1.1 that affects RankD plots
        # See #912 and #914 for details.
        if mpl.__version__ == "3.1.1":
            msg = (
                "RankD plots may be clipped when using matplotlib v3.1.1, "
                "upgrade to matplotlib v3.1.2 or later to fix the plots."
            )
            warnings.warn(msg, YellowbrickWarning)

        # Set the title for all RankD visualizations.
        self.set_title(
            "{} Ranking of {} Features".format(
                self.ranking_.title(), len(self.features_)
            )
        )


##########################################################################
## Rank 1D Feature Visualizer
##########################################################################


class Rank1D(RankDBase):
    """
    Rank1D computes a score for each feature in the data set with a specific
    metric or algorithm (e.g. Shapiro-Wilk) then returns the features ranked
    as a bar plot.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    algorithm : one of {'shapiro', }, default: 'shapiro'
        The ranking algorithm to use, default is 'Shapiro-Wilk.

    features : list
        A list of feature names to use.
        If a DataFrame is passed to fit and features is None, feature
        names are selected as the columns of the DataFrame.

    orient : 'h' or 'v', default='h'
        Specifies a horizontal or vertical bar chart.

    show_feature_names : boolean, default: True
        If True, the feature names are used to label the x and y ticks in the
        plot.

    color: string
        Specify color for barchart

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    ranks_ : ndarray
        An array of rank scores with shape (n,), where n is the
        number of features. It is computed during `fit`.

    Examples
    --------
    >>> visualizer = Rank1D()
    >>> visualizer.fit(X, y)
    >>> visualizer.transform(X)
    >>> visualizer.show()
    """

    ranking_methods = {"shapiro": lambda X: np.array([shapiro(x)[0] for x in X.T])}

    def __init__(
        self,
        ax=None,
        algorithm="shapiro",
        features=None,
        orient="h",
        show_feature_names=True,
        color=None,
        **kwargs
    ):
        """
        Initialize the class with the options required to rank and
        order features as well as visualize the result.
        """
        super(Rank1D, self).__init__(
            ax=ax,
            algorithm=algorithm,
            features=features,
            show_feature_names=show_feature_names,
            **kwargs
        )
        self.color = color
        self.orientation_ = orient

    def draw(self, **kwargs):
        """
        Draws the bar plot of the ranking array of features.
        """
        if self.orientation_ == "h":
            # Make the plot
            self.ax.barh(np.arange(len(self.ranks_)), self.ranks_, color=self.color)

            # Add ticks and tick labels
            self.ax.set_yticks(np.arange(len(self.ranks_)))
            if self.show_feature_names_:
                self.ax.set_yticklabels(self.features_)
            else:
                self.ax.set_yticklabels([])

            # Order the features from top to bottom on the y axis
            self.ax.invert_yaxis()

            # Turn off y grid lines
            self.ax.yaxis.grid(False)

        elif self.orientation_ == "v":
            # Make the plot
            self.ax.bar(np.arange(len(self.ranks_)), self.ranks_, color=self.color)

            # Add ticks and tick labels
            self.ax.set_xticks(np.arange(len(self.ranks_)))
            if self.show_feature_names_:
                self.ax.set_xticklabels(self.features_, rotation=90)
            else:
                self.ax.set_xticklabels([])

            # Turn off x grid lines
            self.ax.xaxis.grid(False)

        else:
            raise YellowbrickValueError("Orientation must be 'h' or 'v'")


##########################################################################
## Rank 2D Feature Visualizer
##########################################################################


class Rank2D(RankDBase):
    """
    Rank2D performs pairwise comparisons of each feature in the data set with
    a specific metric or algorithm (e.g. Pearson correlation) then returns
    them ranked as a lower left triangle diagram.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    algorithm : str, default: 'pearson'
        The ranking algorithm to use, one of: 'pearson', 'covariance', 'spearman',
        or 'kendalltau'.

    features : list
        A list of feature names to use.
        If a DataFrame is passed to fit and features is None, feature
        names are selected as the columns of the DataFrame.

    colormap : string or cmap, default: 'RdBu_r'
        optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or colormap to
        color them on a continuous scale.

    show_feature_names : boolean, default: True
        If True, the feature names are used to label the axis ticks in the plot.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    ranks_ : ndarray
        An array of rank scores with shape (n,n), where n is the
        number of features. It is computed during `fit`.

    Examples
    --------

    >>> visualizer = Rank2D()
    >>> visualizer.fit(X, y)
    >>> visualizer.transform(X)
    >>> visualizer.show()

    Notes
    -----
    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """

    ranking_methods = {
        "pearson": lambda X: np.corrcoef(X.transpose()),
        "covariance": lambda X: np.cov(X.transpose()),
        "spearman": lambda X: spearmanr(X, axis=0)[0],
        "kendalltau": lambda X: kendalltau(X),
    }

    def __init__(
        self,
        ax=None,
        algorithm="pearson",
        features=None,
        colormap="RdBu_r",
        show_feature_names=True,
        **kwargs
    ):
        """
        Initialize the class with the options required to rank and
        order features as well as visualize the result.
        """
        super(Rank2D, self).__init__(
            ax=ax,
            algorithm=algorithm,
            features=features,
            show_feature_names=show_feature_names,
            **kwargs
        )
        self.colormap = colormap

    def draw(self, **kwargs):
        """
        Draws the heatmap of the ranking matrix of variables.
        """
        # Set the axes aspect to be equal
        self.ax.set_aspect("equal")

        # Generate a mask for the upper triangle
        mask = np.zeros_like(self.ranks_, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Draw the heatmap
        # TODO: Move mesh to a property so the colorbar can be finalized
        data = np.ma.masked_where(mask, self.ranks_)
        mesh = self.ax.pcolormesh(data, cmap=self.colormap, vmin=-1, vmax=1)

        # Set the Axis limits
        self.ax.set(xlim=(0, data.shape[1]), ylim=(0, data.shape[0]))

        # Add the colorbar
        cb = self.ax.figure.colorbar(mesh, None, self.ax)
        cb.outline.set_linewidth(0)

        # Reverse the rows to get the lower left triangle
        self.ax.invert_yaxis()

        # Add ticks and tick labels
        self.ax.set_xticks(np.arange(len(self.ranks_)) + 0.5)
        self.ax.set_yticks(np.arange(len(self.ranks_)) + 0.5)
        if self.show_feature_names_:
            self.ax.set_xticklabels(self.features_, rotation=90)
            self.ax.set_yticklabels(self.features_)
        else:
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])


##########################################################################
## Quick Methods
##########################################################################

def rank1d(
    X,
    y=None,
    ax=None,
    algorithm="shapiro",
    features=None,
    orient="h",
    show_feature_names=True,
    color=None,
    show=True,
    **kwargs
):
    """Scores each feature with the algorithm and ranks them in a bar plot.

    This helper function is a quick wrapper to utilize the Rank1D Visualizer
    (Transformer) for one-off analysis.

    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features

    y : ndarray or Series of length n
        An array or series of target or class values

    ax : matplotlib axes
        the axis to plot the figure on.

    algorithm : one of {'shapiro', }, default: 'shapiro'
        The ranking algorithm to use, default is 'Shapiro-Wilk.

    features : list
        A list of feature names to use.
        If a DataFrame is passed to fit and features is None, feature
        names are selected as the columns of the DataFrame.

    orient : 'h' or 'v'
        Specifies a horizontal or vertical bar chart.

    show_feature_names : boolean, default: True
        If True, the feature names are used to label the axis ticks in the
        plot.

    color: string
        Specify color for barchart

    show: bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot
        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply
        calls ``finalize()``

     kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Returns
    -------
    viz : Rank1D
        Returns the fitted, finalized visualizer.

    """
    # Instantiate the visualizer
    visualizer = Rank1D(
        ax=ax,
        algorithm=algorithm,
        features=features,
        orient=orient,
        show_feature_names=show_feature_names,
        color=color,
        **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y)
    visualizer.transform(X)

    if show:
        visualizer.show()
    else:
        visualizer.finalize()

    # Return the visualizer object
    return visualizer


def rank2d(
    X,
    y=None,
    ax=None,
    algorithm="pearson",
    features=None,
    colormap="RdBu_r",
    show_feature_names=True,
    show=True,
    **kwargs
):
    """Rank2D quick method

    Rank2D performs pairwise comparisons of each feature in the data set with
    a specific metric or algorithm (e.g. Pearson correlation) then returns
    them ranked as a lower left triangle diagram.

    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features to perform the pairwise compairsons on.

    y : ndarray or Series of length n, default: None
        An array or series of target or class values, optional (not used).

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    algorithm : str, default: 'pearson'
        The ranking algorithm to use, one of: 'pearson', 'covariance', 'spearman',
        or 'kendalltau'.

    features : list
        A list of feature names to use.
        If a DataFrame is passed to fit and features is None, feature names are
        selected as the columns of the DataFrame.

    colormap : string or cmap, default: 'RdBu_r'
        optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or colormap to
        color them on a continuous scale.

    show_feature_names : boolean, default: True
        If True, the feature names are used to label the axis ticks in the plot.

    show: bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot
        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply
        calls ``finalize()``

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Returns
    -------
    viz : Rank2D
        Returns the fitted, finalized visualizer that created the Rank2D heatmap.
    """
    # Instantiate the visualizer
    viz = Rank2D(
        ax=ax,
        algorithm=algorithm,
        features=features,
        colormap=colormap,
        show_feature_names=show_feature_names,
        **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    viz.fit(X, y)
    viz.transform(X)

    # Show or finalize
    if show:
        viz.show()
    else:
        viz.finalize()

    # Return the visualizer object
    return viz
