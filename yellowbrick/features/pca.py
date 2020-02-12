# -*- coding: utf-8 -*-
# yellowbrick.features.pca
# Decomposition based feature visualization with PCA.
#
# Author:   Carlo Morales
# Author:   Raúl Peralta Lozada
# Author:   Benjamin Bengfort
# Created:  Tue May 23 18:34:27 2017 -0400
#
# Copyright (C) 2017 The scikit-yb developers
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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from yellowbrick.style import palettes
from yellowbrick.features.projection import ProjectionVisualizer
from yellowbrick.exceptions import YellowbrickValueError, NotFitted

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA as PCATransformer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError


##########################################################################
# 2D and 3D PCA Visualizer
##########################################################################


class PCA(ProjectionVisualizer):
    """
    Produce a two or three dimensional principal component plot of a data array
    projected onto its largest sequential principal components. It is common
    practice to scale the data array ``X`` before applying a PC decomposition.
    Variable scaling can be controlled using the ``scale`` argument.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in, the current axes
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

    scale : bool, default: True
        Boolean that indicates if user wants to scale data.

    projection : int or string, default: 2
        The number of axes to project into, either 2d or 3d. To plot 3d plots
        with matplotlib, please ensure a 3d axes is passed to the visualizer,
        otherwise one will be created using the current figure.

    proj_features : bool, default: False
        Boolean that indicates if the user wants to project the features
        in the projected space. If True the plot will be similar to a biplot.

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

    alpha : float, default: 0.75
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    random_state : int, RandomState instance or None, optional (default None)
        This parameter sets the random state on this solver. If the input X is
        larger than 500x500 and the number of components to extract is lower
        than 80% of the smallest dimension of the data, then the more efficient
        `randomized` solver is enabled.

    colorbar : bool, default: True
        If the target_type is "continous" draw a colorbar to the right of the
        scatter plot. The colobar axes is accessible using the cax property.

    heatmap : bool, default: False
        Add a heatmap showing contribution of each feature in the principal components.
        Also draws a colorbar for readability purpose. The heatmap is accessible
        using lax property and colorbar using uax property.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    pca_components_ : ndarray, shape (n_features, n_components)
        This tells about the magnitude of each feature in the pricipal components.
        This is primarily used to draw the biplots.

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
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> visualizer = PCA()
    >>> visualizer.fit_transform(X, y)
    >>> visualizer.show()

    """

    def __init__(
        self,
        ax=None,
        features=None,
        classes=None,
        scale=True,
        projection=2,
        proj_features=False,
        colors=None,
        colormap=None,
        alpha=0.75,
        random_state=None,
        colorbar=True,
        heatmap=False,
        **kwargs
    ):
        super(PCA, self).__init__(
            ax=ax,
            features=features,
            classes=classes,
            colors=colors,
            colormap=colormap,
            projection=projection,
            alpha=alpha,
            colorbar=colorbar,
            **kwargs
        )

        # Data Parameters
        self.scale = scale
        self.proj_features = proj_features

        # Create the PCA transformer
        self.pca_transformer = Pipeline(
            [
                ("scale", StandardScaler(with_std=self.scale)),
                ("pca", PCATransformer(self.projection, random_state=random_state)),
            ]
        )
        self.alpha = alpha

        # Visual Parameters
        self.heatmap = heatmap

        self._uax, self._lax = None, None

        # No heatmap can be drawn with 3d plots as they do not have permit axes
        # division.
        if self.projection == 3 and self.heatmap:
            raise YellowbrickValueError(
                "heatmap and colorbar are not compatible with 3d projections"
            )

    @property
    def uax(self):
        """
        The axes of the colorbar, bottom of scatter plot. This is the colorbar
        for heatmap and not for the scatter plot.
        """
        if self._uax is None:
            raise AttributeError("This visualizer does not have an axes for colorbar")

        return self._uax

    @property
    def lax(self):
        """
        The axes of the heatmap below scatter plot.
        """
        if self._lax is None:
            raise AttributeError("This visualizer does not have an axes for heatmap")

        return self._lax

    def layout(self, divider=None):
        """
        Creates the layout for colorbar and heatmap, adding new axes for the heatmap
        if necessary and modifying the aspect ratio. Does not modify the axes or the
        layout if ``self.heatmap`` is ``False`` or ``None``.

        Parameters
        ----------
        divider: AxesDivider
            An AxesDivider to be passed among all layout calls.
        """

        # Ensure matplotlib version compatibility
        if make_axes_locatable is None:
            raise YellowbrickValueError(
                (
                    "heatmap requires matplotlib 2.0.2 or greater "
                    "please upgrade matplotlib or set heatmap=False on the visualizer"
                )
            )

        # Create the new axes for the colorbar and heatmap
        if divider is None:
            divider = make_axes_locatable(self.ax)

        # Call to super class ensures that a colorbar is drawn when target is
        # continuous.
        super(PCA, self).layout(divider)

        if self.heatmap:

            # Axes for colorbar(for heatmap).
            if self._uax is None:
                self._uax = divider.append_axes("bottom", size="10%", pad=0.7)

            # Axes for heatmap
            if self._lax is None:
                self._lax = divider.append_axes("bottom", size="15%", pad=0.5)

    def fit(self, X, y=None, **kwargs):
        """
        Fits the PCA transformer, transforms the data in X, then draws the
        decomposition in either 2D or 3D space as a scatter plot.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features.

        y : ndarray or Series of length n
            An array or series of target or class values.

        Returns
        -------
        self : visualizer
            Returns self for use in Pipelines.


        """
        # Call super fit to compute features, classes, colors, etc.
        super(PCA, self).fit(X=X, y=y, **kwargs)
        self.pca_transformer.fit(X)
        self.pca_components_ = self.pca_transformer.named_steps["pca"].components_
        return self

    def transform(self, X, y=None, **kwargs):
        """
        Calls the internal `transform` method of the scikit-learn PCA transformer, which
        performs a dimensionality reduction on the input features ``X``. Next calls the
        ``draw`` method of the Yellowbrick visualizer, finally returning a new array of
        transformed features of shape ``(len(X), projection)``.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features.

        y : ndarray or Series of length n
            An array or series of target or class values.

        Returns
        -------
        Xp : ndarray or DataFrame of shape n x m
            Returns a new array-like object of transformed features of shape
            ``(len(X), projection)``.
        """
        try:
            Xp = self.pca_transformer.transform(X)
            self.draw(Xp, y)
            return Xp
        except NotFittedError:
            raise NotFitted.from_estimator(self, "transform")

    def draw(self, Xp, y):
        """
        Plots a scatterplot of points that represented the decomposition,
        `pca_features_`, of the original features, `X`, projected into either 2 or
        3 dimensions.

        If 2 dimensions are selected, a colorbar and heatmap can also be optionally
        included to show the magnitude of each feature value to the component.

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
        # Call to super draw which draws the scatter plot.
        super(PCA, self).draw(Xp, y)
        if self.proj_features:
            # Draws projection features in transformed space.
            self._draw_projection_features(Xp, y)
        if self.projection == 2:
            if self.heatmap:
                if not self.colormap:
                    self.colormap = palettes.DEFAULT_SEQUENCE
                # TODO: change to pcolormesh instead of imshow per #615 spec
                im = self.lax.imshow(
                    self.pca_components_,
                    interpolation="none",
                    cmap=self.colormap,
                    aspect="auto",
                )
                plt.colorbar(
                    im,
                    cax=self.uax,
                    orientation="horizontal",
                    ticks=[self.pca_components_.min(), 0, self.pca_components_.max()],
                )
        return self.ax

    def _draw_projection_features(self, Xp, y):
        """
        Draw the projection of features in the transformed space.
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

        x_vector = self.pca_components_[0]
        y_vector = self.pca_components_[1]
        max_x = max(Xp[:, 0])
        max_y = max(Xp[:, 1])
        if self.projection == 2:
            for i in range(self.pca_components_.shape[1]):
                self.ax.arrow(
                    x=0,
                    y=0,
                    dx=x_vector[i] * max_x,
                    dy=y_vector[i] * max_y,
                    color="r",
                    head_width=0.05,
                    width=0.005,
                )
                self.ax.text(
                    x_vector[i] * max_x * 1.05,
                    y_vector[i] * max_y * 1.05,
                    self.features_[i],
                    color="r",
                )
        elif self.projection == 3:
            z_vector = self.pca_components_[2]
            max_z = max(Xp[:, 1])
            for i in range(self.pca_components_.shape[1]):
                self.ax.plot(
                    [0, x_vector[i] * max_x],
                    [0, y_vector[i] * max_y],
                    [0, z_vector[i] * max_z],
                    color="r",
                )
                self.ax.text(
                    x_vector[i] * max_x * 1.05,
                    y_vector[i] * max_y * 1.05,
                    z_vector[i] * max_z * 1.05,
                    self.features_[i],
                    color="r",
                )
        else:
            raise YellowbrickValueError("Projection dimensions must be either 2 or 3")

        return self.ax

    def finalize(self, **kwargs):
        """
        Draws the title, labels, legends, heatmap, and colorbar as specified by the
        keyword arguments.
        """
        super(PCA, self).finalize()

        self.ax.set_title("Principal Component Plot")
        self.ax.set_xlabel("$PC_1$")
        self.ax.set_ylabel("$PC_2$")
        if self.projection == 3:
            self.ax.set_zlabel("$PC_3$")
        if self.heatmap == True:
            self.lax.set_xticks(np.arange(-0.5, len(self.features_)))
            self.lax.set_xticklabels([])
            # Makes the labels centered.
            self.lax.set_xticks(np.arange(0, len(self.features_)), minor=True)
            self.lax.set_xticklabels(
                self.features_, rotation=90, fontsize=12, minor=True
            )
            self.lax.set_yticks(np.arange(0.5, 2))
            self.lax.set_yticklabels(["$PC_1$", "$PC_2$"], va="bottom", fontsize=10)
        self.fig.tight_layout()


##########################################################################
## Quick Method
##########################################################################


def pca_decomposition(
    X,
    y=None,
    ax=None,
    features=None,
    classes=None,
    scale=True,
    projection=2,
    proj_features=False,
    colors=None,
    colormap=None,
    alpha=0.75,
    random_state=None,
    colorbar=True,
    heatmap=False,
    show=True,
    **kwargs
):

    """
    Produce a two or three dimensional principal component plot of the data array ``X``
    projected onto its largest sequential principal components. It is common practice
    to scale the data array ``X`` before applying a PC decomposition. Variable scaling
    can be controlled using the ``scale`` argument.

    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features.

    y : ndarray or Series of length n
        An array or series of target or class values.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in, the current axes
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

    scale : bool, default: True
        Boolean that indicates if user wants to scale data.

    projection : int or string, default: 2
        The number of axes to project into, either 2d or 3d. To plot 3d plots
        with matplotlib, please ensure a 3d axes is passed to the visualizer,
        otherwise one will be created using the current figure.

    proj_features : bool, default: False
        Boolean that indicates if the user wants to project the features
        in the projected space. If True the plot will be similar to a biplot.

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

    alpha : float, default: 0.75
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    random_state : int, RandomState instance or None, optional (default None)
        This parameter sets the random state on this solver. If the input X is
        larger than 500x500 and the number of components to extract is lower
        than 80% of the smallest dimension of the data, then the more efficient
        `randomized` solver is enabled.

    colorbar : bool, default: True
        If the target_type is "continous" draw a colorbar to the right of the
        scatter plot. The colobar axes is accessible using the cax property.

    heatmap : bool, default: False
        Add a heatmap showing contribution of each feature in the principal components.
        Also draws a colorbar for readability purpose. The heatmap is accessible
        using lax property and colorbar using uax property.

    show : bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot
        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply
        calls ``finalize()``

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    pca_components_ : ndarray, shape (n_features, n_components)
        This tells about the magnitude of each feature in the pricipal components.
        This is primarily used to draw the biplots.

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
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> pca_decomposition(X, y, colors=['r', 'g', 'b'], projection=3)

    """
    # Instantiate the visualizer
    visualizer = PCA(
        ax=ax,
        features=features,
        classes=classes,
        scale=scale,
        projection=projection,
        proj_features=proj_features,
        colors=colors,
        colormap=colormap,
        alpha=alpha,
        random_state=random_state,
        colorbar=colorbar,
        heatmap=heatmap,
        **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y)
    visualizer.transform(X, y)

    if show:
        visualizer.show()
    else:
        visualizer.finalize()

    # Returns the visualizer object.
    return visualizer


# Alias for PCA
PCADecomposition = PCA
