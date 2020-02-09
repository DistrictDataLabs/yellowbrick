# yellowbrick.features.manifold
# Use manifold algorithms for high dimensional visualization.
#
# Author:  Benjamin Bengfort
# Created: Sat May 12 11:25:24 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: manifold.py [02f8c27] benjamin@bengfort.com $

"""
Use manifold algorithms for high dimensional visualization.
"""

##########################################################################
## Imports
##########################################################################

import warnings

from yellowbrick.utils.timer import Timer
from yellowbrick.utils.types import is_estimator
from yellowbrick.exceptions import ModelError, NotFitted
from yellowbrick.features.projection import ProjectionVisualizer
from yellowbrick.exceptions import YellowbrickValueError, YellowbrickWarning

from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap, MDS, TSNE, SpectralEmbedding


##########################################################################
## Supported manifold algorithms by name lookup
##########################################################################

MANIFOLD_ALGORITHMS = {
    "lle": LocallyLinearEmbedding(method="standard", eigen_solver="auto"),
    "ltsa": LocallyLinearEmbedding(method="ltsa", eigen_solver="auto"),
    "hessian": LocallyLinearEmbedding(method="hessian", eigen_solver="auto"),
    "modified": LocallyLinearEmbedding(method="modified", eigen_solver="auto"),
    "isomap": Isomap(),
    "mds": MDS(),
    "spectral": SpectralEmbedding(),
    "tsne": TSNE(init="pca"),
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


##########################################################################
## Manifold Embeddings
##########################################################################


class Manifold(ProjectionVisualizer):
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

    manifold : str or Transformer, default: "mds"
        Specify the manifold algorithm to perform the embedding. Either one of
        the strings listed in the table above, or an actual scikit-learn
        transformer. The constructed manifold is accessible with the manifold
        property, so as to modify hyperparameters before fit.

    n_neighbors : int, default: None
        Many manifold algorithms are nearest neighbors based, for those that
        are, this parameter specfies the number of neighbors to use in the
        embedding. If n_neighbors is not specified for those embeddings, it is
        set to 5 and a warning is issued. If the manifold algorithm doesn't use
        nearest neighbors, then this parameter is ignored.

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

    random_state : int or RandomState, default: None
        Fixes the random state for stochastic manifold algorithms.

    colorbar : bool, default: True
        If the target_type is "continous" draw a colorbar to the right of the
        scatter plot. The colobar axes is accessible using the cax property.

    kwargs : dict
        Keyword arguments passed to the base class and may influence the
        feature visualization properties.

    Attributes
    ----------
    fit_time_ : yellowbrick.utils.timer.Timer
        The amount of time in seconds it took to fit the Manifold.

    classes_ : ndarray, shape (n_classes,)
        The class labels that define the discrete values in the target. Only
        available if the target type is discrete. This is guaranteed to be
        strings even if the classes are a different type.

    features_ : ndarray, shape (n_features,)
        The names of the features discovered or used in the visualizer that
        can be used as an index to access or modify data in X. If a user passes
        feature names in, those features are used. Otherwise the columns of a
        DataFrame are used or just simply the indices of the data array.

    range_ : (min y, max y)
        A tuple that describes the minimum and maximum values in the target.
        Only available if the target type is continuous.

    Examples
    --------

    >>> viz = Manifold(manifold='isomap', target='discrete')
    >>> viz.fit_transform(X, y)
    >>> viz.show()

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
    .. _`Hessian LLE`: http://scikit-learn.org/stable/modules/manifold.html#hessian-eigenmapping
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
        manifold="mds",
        n_neighbors=None,
        features=None,
        classes=None,
        colors=None,
        colormap=None,
        target_type="auto",
        projection=2,
        alpha=0.75,
        random_state=None,
        colorbar=True,
        **kwargs
    ):

        super(Manifold, self).__init__(
            ax,
            features,
            classes,
            colors,
            colormap,
            target_type,
            projection,
            alpha,
            colorbar,
            **kwargs
        )
        self._name = None
        self._manifold = None

        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.manifold = manifold  # must be set last

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
                    "could not create manifold for '{}'".format(str(transformer))
                )

            # 2 components is required for 2D plots
            n_components = self.projection
            requires_default_neighbors = {
                "lle",
                "ltsa",
                "isomap",
                "hessian",
                "spectral",
                "modified",
            }

            # Check if the n_neighbors attribute needs to be set.
            if self.n_neighbors is None and transformer in requires_default_neighbors:
                if transformer == "hessian":
                    self.n_neighbors = int(
                        1 + (n_components * (1 + (n_components + 1) / 2))
                    )
                else:
                    self.n_neighbors = 5

                # Issue a warning that the n_neighbors was set to a default.
                warnmsg = (
                    "using n_neighbors={};"
                    " please explicitly specify for the '{}' manifold"
                ).format(self.n_neighbors, str(transformer))
                warnings.warn(warnmsg, YellowbrickWarning)

            # Create a new transformer with the specified params
            self._name = MANIFOLD_NAMES[transformer]
            transformer = clone(self.ALGORITHMS[transformer])
            params = {
                "n_components": n_components,
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

    def fit(self, X, y=None, **kwargs):
        """
        Fits the manifold on X and transforms the data to plot it on the axes.
        See fit_transform() for more details.

        Parameters
        ----------
        X : array-like of shape (n, m)
            A matrix or data frame with n instances and m features

        y : array-like of shape (n,), optional
            A vector or series with target values for each instance in X. This
            vector is used to determine the color of the points in X.

        Returns
        -------
        self : Manifold
            Returns the visualizer object.

        """
        if not hasattr(self.manifold, "transform"):
            name = self.manifold.__class__.__name__
            raise ModelError(
                (
                    "{} requires data to be simultaneously fit and transformed, "
                    "use fit_transform instead"
                ).format(name)
            )

        # Call super to compute features, classes, colors, etc.
        super(Manifold, self).fit(X, y)
        with Timer() as self.fit_time_:
            self.manifold.fit(X)
        return self

    def fit_transform(self, X, y=None, **kwargs):
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
            A matrix or data frame with n instances and m features

        y : array-like of shape (n,), optional
            A vector or series with target values for each instance in X. This
            vector is used to determine the color of the points in X.

        Returns
        -------
        Xprime : array-like of shape (n, 2)
            Returns the 2-dimensional embedding of the instances.

        """
        # Because some manifolds do not have transform, we cannot call individual
        # fit and transform methods, but must do it manually here.

        # Call super fit to compute features, classes, colors, etc.
        super(Manifold, self).fit(X, y)
        with Timer() as self.fit_time_:
            Xp = self.manifold.fit_transform(X)
        self.draw(Xp, y)
        return Xp

    def transform(self, X, y=None, **kwargs):
        """
        Returns the transformed data points from the manifold embedding.

        Parameters
        ----------
        X : array-like of shape (n, m)
            A matrix or data frame with n instances and m features

        y : array-like of shape (n,), optional
            The target, used to specify the colors of the points.

        Returns
        -------
        Xprime : array-like of shape (n, 2)
            Returns the 2-dimensional embedding of the instances.

        Note
        ----
        This method does not work with MDS, TSNE and SpectralEmbedding because
        it is yet to be implemented in sklearn.
        """
        # Because some manifolds do not have transform we cannot call super
        try:
            Xp = self.manifold.transform(X)
            self.draw(Xp, y)
            return Xp
        except NotFittedError:
            raise NotFitted.from_estimator(self, "transform")
        except AttributeError:
            name = self.manifold.__class__.__name__
            raise ModelError(
                (
                    "{} requires data to be simultaneously fit and transformed, "
                    "use fit_transform instead"
                ).format(name)
            )

        return Xp

    def draw(self, Xp, y=None):
        # Calls draw method from super class which draws scatter plot.
        super(Manifold, self).draw(Xp, y)
        return self.ax

    def finalize(self):
        """
        Add title and modify axes to make the image ready for display.
        """
        self.set_title(
            "{} Manifold (fit in {:0.2f} seconds)".format(
                self._name, self.fit_time_.interval
            )
        )
        self.ax.set_xlabel("Using {} features".format(len(self.features_)))
        # Draws legend for discrete target and colorbar for continuous.
        super(Manifold, self).finalize()


##########################################################################
## Quick Method
##########################################################################


def manifold_embedding(
    X,
    y=None,
    ax=None,
    manifold="mds",
    n_neighbors=None,
    features=None,
    classes=None,
    colors=None,
    colormap=None,
    target_type="auto",
    projection=2,
    alpha=0.75,
    random_state=None,
    colorbar=True,
    show=True,
    **kwargs
):
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

    ax : matplotlib.Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    manifold : str or Transformer, default: "lle"
        Specify the manifold algorithm to perform the embedding. Either one of
        the strings listed in the table above, or an actual scikit-learn
        transformer. The constructed manifold is accessible with the manifold
        property, so as to modify hyperparameters before fit.

    n_neighbors : int, default: None
        Many manifold algorithms are nearest neighbors based, for those that
        are, this parameter specfies the number of neighbors to use in the
        embedding. If n_neighbors is not specified for those embeddings, it is
        set to 5 and a warning is issued. If the manifold algorithm doesn't use
        nearest neighbors, then this parameter is ignored.

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

    random_state : int or RandomState, default: None
        Fixes the random state for stochastic manifold algorithms.

    colorbar : bool, default: True
        If the target_type is "continous" draw a colorbar to the right of the
        scatter plot. The colobar axes is accessible using the cax property.

    show: bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot
        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply
        calls ``finalize()``

    kwargs : dict
        Keyword arguments passed to the base class and may influence the
        feature visualization properties.

    Returns
    -------
    viz : Manifold
        Returns the fitted, finalized visualizer
    """
    # Instantiate the visualizer
    viz = Manifold(
        ax=ax,
        manifold=manifold,
        n_neighbors=n_neighbors,
        features=features,
        classes=classes,
        colors=colors,
        colormap=colormap,
        target_type=target_type,
        projection=projection,
        alpha=alpha,
        random_state=random_state,
        colorbar=colorbar,
        **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    viz.fit_transform(X, y)

    if show:
        viz.show()
    else:
        viz.finalize()

    # Return the visualizer object
    return viz
