# -*- coding: utf-8 -*-
# yellowbrick.features.pca
# Decomposition based feature visualization with PCA.
#
# Author:   Carlo Morales <@cjmorale>
# Author:   Raúl Peralta Lozada <@RaulPL>
# Author:   Benjamin Bengfort <@bbengfort>
# Created:  Tue May 23 18:34:27 2017 -0400
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

from yellowbrick.features.projection import ProjectionVisualizer
from yellowbrick.style import palettes
from yellowbrick.exceptions import YellowbrickValueError, NotFitted

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError


##########################################################################
# 2D and 3D PCA Visualizer
##########################################################################

class PCADecomposition(ProjectionVisualizer):
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
        A list of feature names to use. If a DataFrame is passed to fit and features 
        is None, feature names are selected as the columns of the DataFrame.

    scale : bool, default: True
        Boolean that indicates if user wants to scale data.

    proj_dim : int, default: 2
        Dimension of the PCA visualizer.

    proj_features : bool, default: False
        Boolean that indicates if the user wants to project the features
        in the projected space. If True the plot will be similar to a biplot.

    color : list or tuple of colors, default: None
        Specify the colors for each individual class.

    colormap : string or cmap, default: None
        Optional string or matplotlib cmap to colorize lines.
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    alpha : float, default: 0.75
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    random_state : int, RandomState instance or None, optional (default None)
        This parameter sets the random state on this solver. If the input X is 
        larger than 500x500 and the number of components to extract is lower 
        than 80% of the smallest dimension of the data, then the more efficient 
        `randomized` solver is enabled.

    colorbar : bool, default: False
        Add a colorbar to shows the range in magnitude of feature values to the
        component.

    heatmap : bool, default: False
        Add a heatmap to explain which features contribute most to which component.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> visualizer = PCADecomposition()
    >>> visualizer.fit_transform(X, y)
    >>> visualizer.poof()

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
        colormap=palettes.DEFAULT_SEQUENCE,
        alpha=0.75,
        random_state=None,
        colorbar=True,
        heatmap=False,
        **kwargs
    ):
        super(PCADecomposition, self).__init__(ax=ax, features=features, 
             classes=classes, colors=colors, colormap=colormap, projection=projection, 
             alpha=alpha, **kwargs)

        # Data Parameters
        self.scale = scale
        self.proj_features = proj_features

        # Create the PCA transformer
        self.pca_transformer = Pipeline(
            [
                ("scale", StandardScaler(with_std=self.scale)),
                ("pca", PCA(self.projection, random_state=random_state)),
            ]
        )
        self.alpha = alpha

        # Visual Parameters
        
        self.colorbar = colorbar
        self.heatmap = heatmap

        self._uax, self._lax = None, None

        if self.projection == 3 and self.heatmap:
            raise YellowbrickValueError(
                "heatmap and colorbar are not compatible with 3d projections"
            )

    @property
    def uax(self):
        """
        The axes of the colorbar, right of the scatterplot.
        """
        if self._uax is None:
            raise AttributeError(
                "This visualizer does not have an axes for colorbar"
            )

        return self._uax
    
    @property
    def lax(self):
        """
        The axes of the colorbar, right of the scatterplot.
        """
        if self._lax is None:
            raise AttributeError(
                "This visualizer does not have an axes for heatmap"
            )

        return self._lax

    def layout(self, divider=None):
        """
        Creates the layout for colorbar and heatmap, adding new axes for the heatmap
        if necessary and modifying the aspect ratio. Does not modify the axes or the
        layout if ``self.heatmap`` is ``False`` or ``None``.
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
        
        super(PCADecomposition, self).layout(divider)

        

        if self.heatmap:
            if self._uax is None:
                self._uax = divider.append_axes("bottom", size="20%", pad=0.7)

            if self._lax is None:
                self._lax = divider.append_axes("bottom", size="100%", pad=0.5)

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
            Returns self for use in Pipelines
        """
        super(PCADecomposition, self).fit(X=X, y=y, **kwargs)
        self.pca_transformer.fit(X)
        self.pca_components_ = self.pca_transformer.named_steps["pca"].components_
        return self

    def transform(self, X, y=None, **kwargs):
        """
        Calls the internal `transform` method of the scikit-learn PCA transformer, which
        performs a dimensionality reduction on the input features ``X``. Next calls the
        ``draw`` method of the Yellowbrick visualizer, finally returning a new array of
        transformed features of shape ``(len(X), proj_dim)``.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features.

        y : ndarray or Series of length n
            An array or series of target or class values.

        Returns
        -------
        pca_features_ : ndarray or DataFrame of shape n x m
            Returns a new array-like object of transformed features of shape
            ``(len(X), proj_dim)``.
        """
        try:
            Xp = self.pca_transformer.transform(X)
            self.draw(Xp, y)
            return Xp
        except NotFittedError:
            raise NotFitted.from_estimator(self, 'transform')
            
        
    def draw(self, X, y):
        """
        Plots a scatterplot of points that represented the decomposition,
        `pca_features_`, of the original features, `X`, projected into either 2 or
        3 dimensions.

        If 2 dimensions are selected, a colorbar and heatmap can also be optionally
        included to show the magnitude of each feature value to the component.

        Returns
        -------
        self : visualizer.ax
            Returns the axes of the visualizer for use in Pipelines
        """
        super(PCADecomposition, self).draw(X, y)
        if self.proj_features:
            self.draw_projection_features(X, y)
        if self.projection == 2:
            if self.heatmap:
                # TODO: change to pcolormesh instead of imshow per #615 spec
                im = self.lax.imshow(
                    self.pca_components_, interpolation="none", cmap=self.colormap
                )
                plt.colorbar(
                    im,
                    cax=self.uax,
                    orientation="horizontal",
                    ticks=[self.pca_components_.min(), 0, self.pca_components_.max()],
                )
        return self.ax

    def draw_projection_features(self, X, y):
        
        x_vector = self.pca_components_[0]
        y_vector = self.pca_components_[1]
        max_x = max(X[:, 0])
        max_y = max(X[:, 1])
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
            max_z = max(X[:, 1])
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
        super(PCADecomposition, self).finalize()
        
        self.ax.set_title("Principal Component Plot")
        self.ax.set_xlabel("Principal Component 1", linespacing=1)
        self.ax.set_ylabel("Principal Component 2", linespacing=1.2)
        if self.projection == 3:
            self.ax.set_zlabel("Principal Component 3", linespacing=1.2)
        if self.heatmap == True:
            self.lax.set_xticks(np.arange(-0.5, len(self.features_)))
            self.lax.set_xticklabels(
                self.features_, rotation=90, ha="left", fontsize=12
            )
            self.lax.set_yticks(np.arange(0.5, 2))
            self.lax.set_yticklabels(
                ["First PC", "Second PC"], va="bottom", fontsize=12
            )

        



##########################################################################
## Quick Method
##########################################################################


def pca_decomposition(
    X,
    y=None,
    ax=None,
    features=None,
    scale=True,
    proj_dim=2,
    proj_features=False,
    color=None,
    colormap=palettes.DEFAULT_SEQUENCE,
    alpha=0.75,
    random_state=None,
    colorbar=False,
    heatmap=False,
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
        A list of feature names to use. If a DataFrame is passed to fit and 
        features is None, feature names are selected as the columns of the DataFrame.

    scale : bool, default: True
        Boolean that indicates if user wants to scale data.

    proj_dim : int, default: 2
        Dimension of the PCA visualizer.

    proj_features : bool, default: False
        Boolean that indicates if the user wants to project the features
        in the projected space. If True the plot will be similar to a biplot.

    color : list or tuple of colors, default: None
        Specify the colors for each individual class.

    colormap : string or cmap, default: None
        Optional string or matplotlib cmap to colorize lines.
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    alpha : float, default: 0.75
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    random_state : int, RandomState instance or None, optional (default None)
        If input X is larger than 500x500 and the number of components to
        extract is lower than 80% of the smallest dimension of the data, then
        the more efficient `randomized` solver is enabled, this parameter sets
        the random state on this solver.

    colorbar : bool, default: False
        Add a colorbar to shows the range in magnitude of feature values to the
        component.

    heatmap : bool, default: False
        Add a heatmap to explain which features contribute most to which component.

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
    visualizer = PCADecomposition(
        ax=ax,
        features=features,
        scale=scale,
        proj_dim=proj_dim,
        proj_features=proj_features,
        color=color,
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
    visualizer.poof()

    # Return the axes object on the visualizer
    return visualizer.ax
