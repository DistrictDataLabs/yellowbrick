# yellowbrick.features.manifold
# Use manifold algorithms for high dimensional visualization.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Sat May 12 11:25:24 2018 -0400
#
# ID: manifold.py [] benjamin@bengfort.com $

"""
Use manifold algorithms for high dimensional visualization.
"""

##########################################################################
## Imports
##########################################################################

import time
import numpy as np
import matplotlib.pyplot as plt

from six import string_types
from matplotlib import patches

from yellowbrick.utils.types import is_estimator
from yellowbrick.style import palettes, resolve_colors
from yellowbrick.features.base import FeatureVisualizer
from yellowbrick.exceptions import YellowbrickValueError, NotFitted

from sklearn.base import clone
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap, MDS, TSNE, SpectralEmbedding


##########################################################################
## Supported manifold algorithms by name lookup
##########################################################################

MANIFOLD_ALGORITHMS = {
    "lle": LocallyLinearEmbedding(method="standard", eigen_solver='auto'),
    "ltsa":LocallyLinearEmbedding(method="ltsa", eigen_solver='auto'),
    "hessian": LocallyLinearEmbedding(method="hessian", eigen_solver='auto'),
    "modified": LocallyLinearEmbedding(method="modified", eigen_solver='auto'),
    "isomap": Isomap(),
    "mds": MDS(),
    "spectral": SpectralEmbedding(),
    "tsne": TSNE(init='pca'),
}

MANIFOLD_NAMES = {
    "lle": "Locally Linear Embedding",
    "ltsa": "LTSA LLE",
    "hessian": "Hessian LLE",
    "modified": "Modified LLE",
    "isomap": "Isomap",
    "mds": "MDS",
    "spectral": "Spectral Embedding",
    "tsne": "t-SNE",
}

# Target type constants
AUTO = "auto"
SINGLE = "single"
DISCRETE = "discrete"
CONTINUOUS = "continuous"


##########################################################################
## Manifold Embeddings
##########################################################################

class Manifold(FeatureVisualizer):
    """
    The Manifold visualizer provides high dimensional visualization for feature
    analysis by embedding data into 2 dimensions using the sklearn.manifold
    package for manifold learning. In brief, manifold learning algorithms are
    unsuperivsed approaches to non-linear dimensionality reduction (unlike PCA
    or SVD) that help visualize latent structures in data.

    The manifold algorithm used to do the embedding in scatter plot space can
    either be a transformer or a string representing one of the already
    specified manifolds as follows:

        ============== ==========================
        Manifold       Description
        -------------- --------------------------
        ``"lle"``      `Locally Linear Embedding`_
        ``"ltsa"``     `LTSA LLE`_
        ``"hessian"``  `Hessian LLE`_
        ``"modified"`` `Modified LLE`_
        ``"isomap"``   `Isomap`_
        ``"mds"``      `Multi-Dimensional Scaling`_
        ``"spectral"`` `Spectral Embedding`_
        ``"tsne"``     `t-SNE`_
        ============== ==========================

    Each of these algorithms embeds non-linear relationships in different ways,
    allowing for an exploration of various structures in the feature space.
    Note however, that each of these algorithms has different time, memory and
    complexity requirements; take special care when using large datasets!

    The Manifold visualizer also shows the specified target (if given) as the
    color of the scatter plot. If a classification or clustering target is
    given, then discrete colors will be used with a legend. If a regression or
    continuous target is specified, then a colormap and colorbar will be shown.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None, the current axes will be used
        or generated if required.

    manifold : str or Transformer, default: "lle"
        Specify the manifold algorithm to perform the embedding. Either one of
        the strings listed in the table above, or an actual scikit-learn
        transformer. The constructed manifold is accessible with the manifold
        property, so as to modify hyperparameters before fit.

    n_neighbors : int, default: 10
        Many manifold algorithms are nearest neighbors based, for those that
        are, this parameter specfies the number of neighbors to use in the
        embedding. If the manifold algorithm doesn't use nearest neighbors,
        then this parameter is ignored.

    colors : str or list of colors, default: None
        Specify the colors used, though note that the specification depends
        very much on whether the target is continuous or discrete. If
        continuous, colors must be the name of a colormap. If discrete, then
        colors can be the name of a palette or a list of colors to use for each
        class in the target.

    target : str, default: "auto"
        Specify the type of target as either "discrete" (classes) or "continuous"
        (real numbers, usually for regression). If "auto", the Manifold will
        attempt to determine the type by counting the number of unique values.

        If the target is discrete, points will be colored by the target class
        and a legend will be displayed. If continuous, points will be displayed
        with a colormap and a color bar will be displayed. In either case, if
        no target is specified, only a single color will be drawn.

    alpha : float, default: 0.7
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    random_state : int or RandomState, default: None
        Fixes the random state for stochastic manifold algorithms.

    kwargs : dict
        Keyword arguments passed to the base class and may influence the
        feature visualization properties.

    Attributes
    ----------
    fit_time_ : float
        The amount of time in seconds it took to fit the Manifold.

    classes_ : np.ndarray, optional
        If discrete, the classes identified in the target y.

    range_ : tuple of floats, optional
        If continuous, the maximum and minimum values in the target y.

    Examples
    --------

    >>> viz = Manifold(manifold='isomap', target='discrete')
    >>> viz.fit_transform(X, y)
    >>> viz.poof()

    Notes
    -----
    Specifying the target as ``'continuous'`` or ``'discrete'`` will influence
    how the visualizer is finally displayed, don't rely on the automatic
    determination from the Manifold!

    Scaling your data with the standard scalar before applying it to the
    visualizer is a great way of increasing performance. Additionally using
    the ``SelectKBest`` transformer may also improve performance and lead to
    better visualizations.

    .. warning::
        Manifold visualizers have extremly varying time, resource, and
        complexity requirements. Sampling data or features may be necessary
        in order to finish a manifold computation.

    .. seealso::
        The Scikit-Learn discussion on `Manifold Learning <http://scikit-learn.org/stable/modules/manifold.html>`_.

    .. _`Locally Linear Embedding`: http://scikit-learn.org/stable/modules/manifold.html#locally-linear-embedding
    .. _`LTSA LLE`: http://scikit-learn.org/stable/modules/manifold.html#local-tangent-space-alignment
    .. _`Hessian LLE`: http://scikit-learn.org/stable/modules/manifold.html#hessian-eigenmapping>
    .. _`Modified LLE`: http://scikit-learn.org/stable/modules/manifold.html#modified-locally-linear-embedding
    .. _`Isomap`: http://scikit-learn.org/stable/modules/manifold.html#isomap
    .. _`Multi-Dimensional Scaling`: http://scikit-learn.org/stable/modules/manifold.html#multi-dimensional-scaling-mds
    .. _`Spectral Embedding`: http://scikit-learn.org/stable/modules/manifold.html#spectral-embedding
    .. _`t-SNE`: http://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne
    """

    ALGORITHMS = MANIFOLD_ALGORITHMS

    def __init__(
        self,
        ax=None,
        manifold="lle",
        n_neighbors=10,
        colors=None,
        target=AUTO,
        alpha=0.7,
        random_state=None,
        **kwargs
    ):
        super(Manifold, self).__init__(ax, **kwargs)
        self._name = None
        self._manifold = None
        self._target_color_type = None

        self.n_neighbors = n_neighbors
        self.colors = colors
        self.target = target
        self.alpha = alpha
        self.random_state = random_state
        self.manifold = manifold # must be set last

    @property
    def manifold(self):
        """
        Property containing the manifold transformer constructed from the
        supplied hyperparameter. Use this property to modify the manifold
        before fit with ``manifold.set_params()``.
        """
        return self._manifold

    @manifold.setter
    def manifold(self, transformer):
        """
        Creates the manifold estimator if a string value is passed in,
        validates other objects passed in.
        """
        if not is_estimator(transformer):
            if transformer not in self.ALGORITHMS:
                raise YellowbrickValueError(
                    "could not create manifold for '%s'".format(str(transformer))
                )

            # Create a new transformer with the specified params
            self._name = MANIFOLD_NAMES[transformer]
            transformer = clone(self.ALGORITHMS[transformer])
            params = {
                "n_components": 2,
                "n_neighbors": self.n_neighbors,
                "random_state": self.random_state,
            }

            for param in list(params.keys()):
                if param not in transformer.get_params():
                    del params[param]

            transformer.set_params(**params)

        self._manifold = transformer
        if self._name is None:
            self._name = self._manifold.__class__.__name__

    def fit(self, X, y=None):
        """
        Fits the manifold on X and transforms the data to plot it on the axes.
        The optional y specified can be used to declare discrete colors. If
        the target is set to 'auto', this method also determines the target
        type, and therefore what colors will be used.

        Note also that fit records the amount of time it takes to fit the
        manifold and reports that information in the visualization.

        Parameters
        ----------
        X : array-like of shape (n, m)
            A matrix or data frame with n instances and m features where m > 2.

        y : array-like of shape (n,), optional
            A vector or series with target values for each instance in X. This
            vector is used to determine the color of the points in X.

        Returns
        -------
        self : Manifold
            Returns the visualizer object.
        """
        # Determine target type
        self._determine_target_color_type(y)

        # Compute classes and colors if target type is discrete
        if self._target_color_type == DISCRETE:
            self.classes_ = np.unique(y)

            color_kwargs = {'n_colors': len(self.classes_)}

            if isinstance(self.colors, string_types):
                color_kwargs['colormap'] = self.colors
            else:
                color_kwargs['colors'] = self.colors

            self._colors = resolve_colors(**color_kwargs)

        # Compute target range if colors are continuous
        elif self._target_color_type == CONTINUOUS:
            y = np.asarray(y)
            self.range_ = (y.min(), y.max())

        start = time.time()
        Xp = self.manifold.fit_transform(X)
        self.fit_time_ = time.time() - start

        self.draw(Xp, y)
        return self

    def transform(self, X):
        """
        Returns the transformed data points from the manifold embedding.

        Parameters
        ----------
        X : array-like of shape (n, m)
            A matrix or data frame with n instances and m features

        Returns
        -------
        Xprime : array-like of shape (n, 2)
            Returns the 2-dimensional embedding of the instances.
        """
        return self.manifold.transform(X)

    def draw(self, X, y=None):
        """
        Draws the points described by X and colored by the points in y. Can be
        called multiple times before finalize to add more scatter plots to the
        axes, however ``fit()`` must be called before use.

        Parameters
        ----------
        X : array-like of shape (n, 2)
            The matrix produced by the ``transform()`` method.

        y : array-like of shape (n,), optional
            The target, used to specify the colors of the points.

        Returns
        -------
        self.ax : matplotlib Axes object
            Returns the axes that the scatter plot was drawn on.
        """
        scatter_kwargs = {"alpha": self.alpha}

        # Determine the colors
        if self._target_color_type == SINGLE:
            scatter_kwargs["c"] = "b"

        elif self._target_color_type == DISCRETE:
            if y is None:
                raise YellowbrickValueError("y is required for discrete target")

            scatter_kwargs["c"] = [
                self._colors[np.searchsorted(self.classes_, (yi))] for yi in y
            ]

        elif self._target_color_type == CONTINUOUS:
            if y is None:
                raise YellowbrickValueError("y is required for continuous target")

            # TODO manually make colorbar so we can draw it in finalize
            scatter_kwargs["c"] = y
            scatter_kwargs["cmap"] = self.colors or palettes.DEFAULT_SEQUENCE

        else:
            # Technically this should never be raised
            raise NotFitted("could not determine target color type")

        # Draw the scatter plot with the associated colors and alpha
        self._scatter = self.ax.scatter(X[:,0], X[:,1], **scatter_kwargs)
        return self.ax

    def finalize(self):
        """
        Add title and modify axes to make the image ready for display.
        """
        self.set_title(
            '{} Manifold (fit in {:0.2f} seconds)'.format(
                self._name, self.fit_time_
            )
        )
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        if self._target_color_type == DISCRETE:
            # Add the legend
            handles = [
                patches.Patch(color=self._colors[idx], label=self.classes_[idx])
                for idx in range(len(self.classes_))
            ]
            self.ax.legend(handles=handles)

        elif self._target_color_type == CONTINUOUS:
            # Add the color bar
            plt.colorbar(self._scatter, ax=self.ax)

    def _determine_target_color_type(self, y):
        """
        Determines the target color type from the vector y as follows:

            - if y is None: only a single color is used
            - if target is auto: determine if y is continuous or discrete
            - otherwise specify supplied target type

        This property will be used to compute the colors for each point.
        """
        if y is None:
            self._target_color_type = SINGLE
        elif self.target == "auto":
            # NOTE: See #73 for a generalization to use when implemented
            if len(np.unique(y)) < 10:
                self._target_color_type = DISCRETE
            else:
                self._target_color_type = CONTINUOUS
        else:
            self._target_color_type = self.target

        if self._target_color_type not in {SINGLE, DISCRETE, CONTINUOUS}:
            raise YellowbrickValueError((
                "could not determine target color type "
                "from target='{}' to '{}'"
            ).format(self.target, self._target_color_type))


##########################################################################
## Quick Method
##########################################################################

def manifold_embedding(
    X,
    y=None,
    ax=None,
    manifold="lle",
    n_neighbors=10,
    colors=None,
    target=AUTO,
    alpha=0.7,
    random_state=None,
    **kwargs):
    """Quick method for Manifold visualizer.

    The Manifold visualizer provides high dimensional visualization for feature
    analysis by embedding data into 2 dimensions using the sklearn.manifold
    package for manifold learning. In brief, manifold learning algorithms are
    unsuperivsed approaches to non-linear dimensionality reduction (unlike PCA
    or SVD) that help visualize latent structures in data.

    .. seealso:: See Manifold for more details.

    Parameters
    ----------
    X : array-like of shape (n, m)
        A matrix or data frame with n instances and m features where m > 2.

    y : array-like of shape (n,), optional
        A vector or series with target values for each instance in X. This
        vector is used to determine the color of the points in X.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None, the current axes will be used
        or generated if required.

    manifold : str or Transformer, default: "lle"
        Specify the manifold algorithm to perform the embedding. Either one of
        the strings listed in the table below, or an actual scikit-learn
        transformer. The constructed manifold is accessible with the manifold
        property, so as to modify hyperparameters before fit.

        ============== ==========================
        Manifold       Description
        -------------- --------------------------
        ``"lle"``      `Locally Linear Embedding <http://scikit-learn.org/stable/modules/manifold.html#locally-linear-embedding>`_
        ``"ltsa"``     `LTSA LLE <http://scikit-learn.org/stable/modules/manifold.html#local-tangent-space-alignment>`_
        ``"hessian"``  `Hessian LLE <http://scikit-learn.org/stable/modules/manifold.html#hessian-eigenmapping>`_
        ``"modified"`` `Modified LLE <http://scikit-learn.org/stable/modules/manifold.html#modified-locally-linear-embedding>`_
        ``"isomap"``   `Isomap <http://scikit-learn.org/stable/modules/manifold.html#isomap>`_
        ``"mds"``      `Multi-Dimensional Scaling <http://scikit-learn.org/stable/modules/manifold.html#multi-dimensional-scaling-mds>`_
        ``"spectral"`` `Spectral Embedding <http://scikit-learn.org/stable/modules/manifold.html#spectral-embedding>`_
        ``"tsne"``     `t-SNE <http://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne>`_
        ============== ==========================

    n_neighbors : int, default: 10
        Many manifold algorithms are nearest neighbors based, for those that
        are, this parameter specfies the number of neighbors to use in the
        embedding. If the manifold algorithm doesn't use nearest neighbors,
        then this parameter is ignored.

    colors : str or list of colors, default: None
        Specify the colors used, though note that the specification depends
        very much on whether the target is continuous or discrete. If
        continuous, colors must be the name of a colormap. If discrete, then
        colors can be the name of a palette or a list of colors to use for each
        class in the target.

    target : str, default: "auto"
        Specify the type of target as either "discrete" (classes) or "continuous"
        (real numbers, usually for regression). If "auto", the Manifold will
        attempt to determine the type by counting the number of unique values.

        If the target is discrete, points will be colored by the target class
        and a legend will be displayed. If continuous, points will be displayed
        with a colormap and a color bar will be displayed. In either case, if
        no target is specified, only a single color will be drawn.

    alpha : float, default: 0.7
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    random_state : int or RandomState, default: None
        Fixes the random state for stochastic manifold algorithms.

    kwargs : dict
        Keyword arguments passed to the base class and may influence the
        feature visualization properties.

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the embedded scatter plot was drawn on.
    """
    # Instantiate the visualizer
    viz = Manifold(
        ax=ax, manifold=manifold, n_neighbors=n_neighbors, colors=colors,
        target=target, alpha = alpha, random_state=random_state, **kwargs
    )

    # Fit and poof (calls draw)
    viz.fit(X, y)
    viz.poof()

    # Return the axes object
    return viz.ax
