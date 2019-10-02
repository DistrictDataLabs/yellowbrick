# yellowbrick.features.decomposition
#
# Author:   George Richardson
# Created:  Fri Mar 2 16:16:00 2018 +0000
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: decomposition.py [0ed6e8a] g.raymond.richardson@gmail.com $

##########################################################################
## Imports
##########################################################################

from yellowbrick.style import palettes
from yellowbrick.features.base import FeatureVisualizer

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

##########################################################################
## Quick Methods
##########################################################################


def explained_variance_visualizer(
    X,
    y=None,
    ax=None,
    scale=True,
    center=True,
    colormap=palettes.DEFAULT_SEQUENCE,
    **kwargs
):
    """Produce a plot of the explained variance produced by a dimensionality
        reduction algorithm using n=1 to n=n_components dimensions. This is a single
        plot to help identify the best trade off between number of dimensions
        and amount of information retained within the data.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n rows with m features

        y : ndarray or Series of length n
            An array or Series of target or class values

        ax : matplotlib Axes, default: None
            The aces to plot the figure on

        scale : bool, default: True
            Boolean that indicates if the values of X should be scaled.

        colormap : string or cmap, default: None
            optional string or matplotlib cmap to colorize lines
            Use either color to colorize the lines on a per class basis or
            colormap to color them on a continuous scale.

        kwargs : dict
            Keyword arguments that are passed to the base class and may influence
            the visualization as defined in other Visualizers.

        Returns
        -------
        viz : ExplainedVariance
            Returns the fitted, finalized visualizer

        Examples
        --------
        >>> from sklearn import datasets
        >>> bc = datasets.load_breast_cancer()
        >>> X = bc = bc.data
        >>> explained_variance_visualizer(X, scale=True, center=True, colormap='RdBu_r')

        """

    # Instantiate the visualizer
    visualizer = ExplainedVariance(X=X)

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y, **kwargs)
    visualizer.transform(X)
    visualizer.finalize()

    # Return the visualizer object
    return visualizer


##########################################################################
## Explained Variance Feature Visualizer
##########################################################################


class ExplainedVariance(FeatureVisualizer):
    """

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The aces to plot the figure on

    scale : bool, default: True
        Boolean that indicates if the values of X should be scaled.

    colormap : string or cmap, default: None
        optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.


    Examples
    --------

    >>> visualizer = ExplainedVariance()
    >>> visualizer.fit(X)
    >>> visualizer.transform(X)
    >>> visualizer.show()

    """

    def __init__(
        self,
        ax=None,
        scale=True,
        center=True,
        n_components=None,
        colormap=palettes.DEFAULT_SEQUENCE,
        **kwargs
    ):

        super(ExplainedVariance, self).__init__(ax=ax, **kwargs)

        self.colormap = colormap
        self.n_components = n_components
        self.center = center
        self.scale = scale
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler(with_mean=self.center, with_std=self.scale)),
                ("pca", PCA(n_components=self.n_components)),
            ]
        )
        self.pca_features = None

    @property
    def explained_variance_(self):
        return self.pipeline.steps[-1][1].explained_variance_

    def fit(self, X, y=None):
        self.pipeline.fit(X)
        self.draw()
        return self

    def transform(self, X):
        self.pca_features = self.pipeline.transform(X)
        return self.pca_features

    def draw(self):
        X = self.explained_variance_
        self.ax.plot(X)
        return self.ax

    def finalize(self, **kwargs):
        # Set the title
        self.set_title("Explained Variance Plot")

        # Set the axes labels
        self.ax.set_ylabel("Explained Variance")
        self.ax.set_xlabel("Number of Components")
