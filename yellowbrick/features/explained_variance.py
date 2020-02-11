# yellowbrick.features.explained_variance
#
# Author:   George Richardson
# Author:   Benjamin Bengfort
# Created:  Fri Mar 2 16:16:00 2018 +0000
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: explained_variance.py [0ed6e8a] g.raymond.richardson@gmail.com $

##########################################################################
## Imports
##########################################################################

from yellowbrick.style import palettes
from yellowbrick.features.base import FeatureVisualizer

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


##########################################################################
## Quick Method
##########################################################################

def explained_variance(
    X,
    y=None,
    ax=None,
    show=True,
    **kwargs
):
    """ExplainedVariance quick method.

    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features to determine principle components for.

    y : ndarray or Series of length n, default: None
        An array or series of target or class values. This argument is not used but is
        enabled for pipeline purposes.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in, the current axes
        will be used (or generated if required).

    show : bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot
        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply
        calls ``finalize()``

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Returns
    -------
    viz : ExplainedVariance
        Returns the fitted, finalized visualizer
    """

    # Instantiate the visualizer
    oz = ExplainedVariance()

    # Fit and transform the visualizer (calls draw)
    oz.fit(X, y)
    oz.transform(X)

    if show:
        oz.show()
    else:
        oz.finalize()

    # Return the visualizer object
    return oz
