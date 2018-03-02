# yellowbrick.features.pca
# Decomposition based feature visualization with PCA.
#
# Author:   Carlo Morales <@cjmorale>
# Created:  Tue May 23 18:34:27 2017 -0400
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: pca.py [] cmorales@pacificmetrics.com $

"""
Decomposition based feature visualization with PCA.
"""

##########################################################################
## Imports
##########################################################################

# NOTE: must import mplot3d to load the 3D projection
import mpl_toolkits.mplot3d # noqa 
import matplotlib.pyplot as plt

from yellowbrick.features.base import FeatureVisualizer
from yellowbrick.style import palettes
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


##########################################################################
## Quick Methods
##########################################################################

def pca_decomposition(X, y=None, ax=None, scale=True, proj_dim=2,
                      colormap=palettes.DEFAULT_SEQUENCE, color=None, **kwargs):
    """Produce a two or three dimensional principal component plot of the data array ``X``
    projected onto it's largest sequential principal components. It is common practice to scale the
    data array ``X`` before applying a PC decomposition. Variable scaling can be controlled using
    the ``scale`` argument.

    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features.

    y : ndarray or Series of length n
        An array or series of target or class values.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on.

    scale : bool, default: True
        Boolean that indicates if the values of X should be scaled.

    proj_dim : int, default: 2
        Dimension of the PCA visualizer.

    colormap : string or cmap, default: None
        Optional string or matplotlib cmap to colorize lines.
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    color : list or tuple of colors, default: None
        Specify the colors for each individual class.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> pca_decomposition(X, color=y, proj_dim=3, colormap='RdBu_r')

    """
    # Instantiate the visualizer
    visualizer = PCADecomposition(X=X, y=y, ax=ax, scale=scale, proj_dim=proj_dim,
                                  colormap= colormap, color=color)

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y, **kwargs)
    visualizer.transform(X)

    # Return the axes object on the visualizer
    return visualizer.poof()

##########################################################################
##2D and #3D PCA Visualizer
##########################################################################

class PCADecomposition(FeatureVisualizer):
    """
    Produce a two or three dimensional principal component plot of the data array ``X``
    projected onto it's largest sequential principal components. It is common practice to scale the
    data array ``X`` before applying a PC decomposition. Variable scaling can be controlled using
    the ``scale`` argument.

    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features.

    y : ndarray or Series of length n
        An array or series of target or class values.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes.
        will be used (or generated if required).

    scale : bool, default: True
        Boolean that indicates if user wants to scale data.

    proj_dim : int, default: 2
        Dimension of the PCA visualizer.

    color : list or tuple of colors, default: None
        Specify the colors for each individual class.

    colormap : string or cmap, default: None
        Optional string or matplotlib cmap to colorize lines.
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> params = {'scale': True, 'center': False, 'col': y}
    >>> visualizer = PCADecomposition(**params)
    >>> visualizer.fit(X)
    >>> visualizer.transform(X)
    >>> visualizer.poof()

    """
    def __init__(self, ax=None, scale=True, color=None, proj_dim=2,
                 colormap=palettes.DEFAULT_SEQUENCE, **kwargs):
        super(PCADecomposition, self).__init__(ax=ax, **kwargs)

        # Data Parameters
        if proj_dim not in (2, 3):
            raise YellowbrickValueError("proj_dim object is not 2 or 3.")

        self.color = color
        self.scale = scale
        self.proj_dim = proj_dim
        self.pca_transformer = Pipeline([('scale', StandardScaler(with_std=self.scale)),
                                         ('pca', PCA(self.proj_dim, ))
                                         ])
        # Visual Parameters
        self.colormap = colormap

    def fit(self, X, y=None, **kwargs):
        self.pca_transformer.fit(X)
        return self

    def transform(self, X, y=None, **kwargs):
        self.pca_features_ = self.pca_transformer.transform(X)
        self.draw()
        return self.pca_features_

    def draw(self, **kwargs):
        X = self.pca_features_

        if self.proj_dim == 2:
            self.ax.scatter(X[:, 0], X[:, 1], c=self.color, cmap=self.colormap)
        if self.proj_dim == 3:
            self.fig = plt.figure()
            self.fig = self.fig.add_subplot(111, projection='3d')

            self.ax = self.fig.scatter(X[:, 0], X[:, 1], X[:, 2], c=self.color,
                                       cmap=self.colormap)
        return self.ax

    def finalize(self, **kwargs):
        # Set the title
        if self.proj_dim == 2:
            self.set_title('Principal Component Plot')
            self.ax.set_ylabel('Principal Component 2')
            self.ax.set_xlabel('Principal Component 1')

        else:
            self.fig.set_title('Principal Component Plot')
            self.fig.set_xlabel('Principal Component 1')
            self.fig.set_ylabel('Principal Component 2')
            self.fig.set_zlabel('Principal Component 3')
