# yellowbrick.features.rankd
# Implements 1D (histograms) and 2D (joint plot) feature rankings.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Oct 07 15:14:01 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: rankd.py [] benjamin@bengfort.com $

"""
Implements 1D (histograms) and 2D (joint plot) feature rankings.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

from yellowbrick.utils import is_dataframe
from yellowbrick.features.base import FeatureVisualizer
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.style.colors import resolve_colors, get_color_cycle


##########################################################################
## Quick Methods
##########################################################################

def rank2d(X, y=None, ax=None, algorithm='pearson', features=None,
           colormap='RdBu_r', **kwargs):
    """Displays pairwise comparisons of features with the algorithm and ranks
    them in a lower-left triangle heatmap plot.

    This helper function is a quick wrapper to utilize the Rank2D Visualizer
    (Transformer) for one-off analysis.

    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features

    y : ndarray or Series of length n
        An array or series of target or class values

    ax : matplotlib axes
        the axis to plot the figure on.

    algorithm : one of {pearson, covariance}
        the ranking algorithm to use, default is Pearson correlation.

    features : list
        a list of feature names to use
        If a DataFrame is passed to fit and features is None, feature
        names are selected as the columns of the DataFrame.

    colormap : string or cmap
        optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the parallel coordinates were drawn on.

    """
    # Instantiate the visualizer
    visualizer = Rank2D(ax, algorithm, features, colormap, **kwargs)

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y, **kwargs)
    visualizer.transform(X)

    # Return the axes object on the visualizer
    return visualizer.ax


##########################################################################
## Rank 2D Feature Visualizer
##########################################################################

class Rank2D(FeatureVisualizer):
    """
    Rank2D performs pairwise comparisons of each feature in the data set with
    a specific metric or algorithm (e.g. Pearson correlation) then returns
    them ranked as a lower left triangle diagram.
    """

    ranking_methods = {
        'pearson': lambda X: np.corrcoef(X.transpose()),
        'covariance': lambda X: np.cov(X.transpose()),
    }

    def __init__(self, ax=None, algorithm='pearson', features=None,
                 colormap='RdBu_r', **kwargs):
        """
        Initialize the Rank2D class with the options required to rank and
        order features as well as visualize the result.

        Parameters
        ----------
        ax : matplotlib axes
            the axis to plot the figure on.

        algorithm : one of {pearson, covariance}
            the ranking algorithm to use, default is Pearson correlation.

        features : list
            a list of feature names to use
            If a DataFrame is passed to fit and features is None, feature
            names are selected as the columns of the DataFrame.

        colormap : string or cmap
            optional string or matplotlib cmap to colorize lines
            Use either color to colorize the lines on a per class basis or
            colormap to color them on a continuous scale.

        kwargs : dict
            keyword arguments passed to the super class.
        """
        super(Rank2D, self).__init__(**kwargs)

        # The figure params
        # TODO: hoist to a higher level base class
        self.ax = ax

        # Data Parameters
        self.ranking_  = algorithm
        self.features_ = features

        # Visual Parameters
        self.colormap = colormap

    def fit(self, X, y=None, **kwargs):
        """
        The fit method gathers information about the state of the visualizer.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs : dict
            Pass generic arguments to the drawing method

        Returns
        ------
        self : instance
            Returns the instance of the transformer/visualizer
        """

        # TODO: This class is identical to the Parallel Coordinates version,
        # so hoist this functionality to a higher level class that is extended
        # by both RadViz and ParallelCoordinates.

        # Get the shape of the data
        nrows, ncols = X.shape

        # Handle the feature names if they're None.
        if self.features_ is None:

            # If X is a data frame, get the columns off it.
            if is_dataframe(X):
                self.features_ = X.columns

            # Otherwise create numeric labels for each column.
            else:
                self.features_ = [
                    str(cdx) for cdx in range(ncols)
                ]

        # Fit always returns self.
        return self


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
        Xp : ndarray
            The transformed matrix, X'
        """
        # Rank and draw the input matrix
        Xp = self.rank(X)
        self.draw(Xp, **kwargs)

        # Return the X matrix, unchanged
        return X

    def rank(self, X, algorithm=None):
        """
        Returns the ranking of each pair of columns as an m by m matrix.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        algorithm : str or None
            The ranking mechanism to use, or None for the default

        Returns
        -------
        R : ndarray
            The mxm ranking matrix of the variables
        """
        algorithm = algorithm or self.ranking_
        algorithm = algorithm.lower()

        if algorithm not in self.ranking_methods:
            raise YellowbrickValueError(
                "'{}' is unrecognized ranking method".format(algorithm)
            )

        return self.ranking_methods[algorithm](X)

    def draw(self, X, **kwargs):
        """
        Draws the heatmap of the ranking matrix of variables.
        """
        # Create the axes if they don't exist
        if self.ax is None:
            self.ax = plt.gca()
            self.ax.set_aspect("equal")

        # Generate a mask for the upper triangle
        mask = np.zeros_like(X, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Reverse the rows to get the lower left triangle
        X = X[::-1]
        mask = mask[::-1]

        # Draw the heatmap
        data = np.ma.masked_where(mask, X)
        mesh = self.ax.pcolormesh(data, cmap=self.colormap, vmin=-1, vmax=1)

        # Set the Axis limits
        self.ax.set(
            xlim=(0, data.shape[1]), ylim=(0, data.shape[0])
        )

        # Add the colorbar
        cb = self.ax.figure.colorbar(mesh, None, self.ax)
        cb.outline.set_linewidth(0)


    def poof(self, outpath=None, **kwargs):
        """
        Display the Rank2D visualization

        Parameters
        ----------
        outpath: path or None
            Save the figure to disk or if None show in a window
        """
        if self.ax is None: return

        # Set the title
        self.ax.set_title(
            "{} Ranking of {} Features".format(
                self.ranking_.title(), len(self.features_)
            )
        )

        if outpath is not None:
            plt.savefig(outpath, **kwargs)
        else:
            plt.show()
