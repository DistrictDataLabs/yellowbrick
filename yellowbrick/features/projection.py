import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


import matplotlib as mpl

from yellowbrick.draw import manual_legend
from yellowbrick.features.base import DataVisualizer, TargetType
from yellowbrick.exceptions import YellowbrickValueError, NotFitted


##########################################################################
## Projection Visualizers
##########################################################################

class ProjectionVisualizer(DataVisualizer):

    """
    Projection Visualizers are the subclass of DataVisualizer which projects
    multi dimensional data to either two or three dimensional space. The transformer
    which converts multi-dimensional data must  be specified in the subclasses.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes.
        will be used (or generated if required).

    features: list, default: None
        a list of feature names to use
        If a DataFrame is passed to fit and features is None, feature
        names are selected as the columns of the DataFrame.

    classes: list, default: None
        a list of class names for the legend
        If classes is None and a y value is passed to fit then the classes
        are selected from the target vector.

    color : list or tuple of colors, default: None
        Specify the colors for each individual class.

    colormap : string or cmap, default: None
        Optional string or matplotlib cmap to colorize points.
        Use either color to colorize the points on a per class basis or
        colormap to color them on a continuous scale.

    target_type : str, default: "auto"
        Specify the type of target as either "discrete" (classes) or "continuous"
        (real numbers, usually for regression). If "auto", then it will
        attempt to determine the type by counting the number of unique values.

    projection : int, default: 2
        Dimension of the Projection Visualizer.

    alpha : float, default: 0.75
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.
    """
    
    def __init__(self, ax=None, features=None, classes=None, color=None,
             colormap=None, target_type="auto", projection=2, alpha=0.75, **kwargs):
        super(ProjectionVisualizer, self).__init__(ax=ax, features=features, 
                                                     classes=classes, color=color,
                                                     colormap=colormap, 
                                                     target_type=target_type, **kwargs)

        if projection not in frozenset((2, 3, '2D', '3D')):
            raise YellowbrickValueError("Projection dimensions must be either 2 or 3")
        if(isinstance(projection,str)):
            projection=np.int(projection[0])
        self.projection = projection
        self.alpha = alpha
        self._cax = None

    @property
    def cax(self):
        """
        The axes of the colorbar, right of the scatterplot.
        """
        if self._cax is None:
            raise AttributeError(
                "This visualizer does not have an axes for colorbar"
            )
        return self._cax       
        
    def _layout(self):
        """
        Creates the layout for colorbar when target type is continuous.
        The colorbar is added to the right of the scatterplot.
        """
        # Ensure matplotlib version compatibility
        if make_axes_locatable is None:
            raise YellowbrickValueError((
                "heatmap requires matplotlib 2.0.2 or greater "
                "please upgrade matplotlib or set heatmap=False on the visualizer"
            ))

        # Create the new axes for the colorbar
        divider = make_axes_locatable(self.ax)
        self._cax = divider.append_axes("right", size="5%", pad=0.3)
        
    def fit(self, X, y=None, **kwargs):
        """
        Fits the transformer on X. Also identifies the classes, features and
        colors by calling the super class.
        
        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix or data frame of n instances with m features where m>2.

        y : array-like of shape (n,), optional
            A vector or series with target values for each instance in X. This
            vector is used to determine the color of the points in X.

        Returns
        -------
        self : instance
            Returns the instance of the transformer/visualizer.
        """
        super(ProjectionVisualizer, self).fit(X, y, **kwargs)
        self.transformer.fit(X)
        return self

    def transform(self, X, y=None, **kwargs):
        """
        Returns the transformed data points from the transformers. It also calls
        draw which draws a scatter plot with transformed data.
        
        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix or data frame of n instances with m features. m must be
            same as that used in ``fit()`` method.

        y : array-like of shape (n,), optional
            A vector or series with target values for each instance in X. This
            vector is used to determine scatter arguments.

        Returns
        -------
        Xprime : array-like of shape (n, 2)
            Returns the 2-dimensional embedding of the instances.
        """
        try:
            Xp = self.transformer.transform(X)
        except AttributeError as e:
            raise AttributeError(str(e) + " try using fit_transform instead.")
        self.draw(Xp, y, **kwargs)
        return Xp
    
    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix or data frame of n instances with m features where m>2.

        y : array-like of shape (n,), optional
            A vector or series with target values for each instance in X. This
            vector is used to determine the color of the points in X.

        Returns
        -------
        Xprime : array-like of shape (n, 2)
            Returns the 2-dimensional embedding of the instances.

        """
        self.fit(X, y, **kwargs).transform(X, y, **kwargs)
    
    def draw(self, X, y=None):
        """
        Draws the points described by X and colored by the points in y. Can be
        called multiple times before finalize to add more scatter plots to the
        axes, however ``fit()`` must be called before use.

        Parameters
        ----------
        X : array-like of shape (n, 2) or (n, 3)
            The matrix produced by the ``transform()`` method.

        y : array-like of shape (n,), optional
            The target, used to specify the colors of the points.

        Returns
        -------
        self.ax : matplotlib Axes object
            Returns the axes that the scatter plot was drawn on.
        """
        scatter_kwargs = self._determine_scatter_kwargs(y);
        
        if self.projection == 2:
            self._scatter = self.ax.scatter(X[:,0], X[:,1], **scatter_kwargs)

        if self.projection == 3:
            self.fig = plt.gcf()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self._scatter = self.ax.scatter(X[:, 0], X[:, 1], X[:, 2], **scatter_kwargs)
        
        return self.ax

    def finalize(self):
        """
        Draws legends and colorbar for scatter plots.
        """
        if self._target_color_type == TargetType.DISCRETE:
            # Add the legend
            manual_legend(self, self.classes_, list(self._colors.values()),
                          frameon=True)

        elif self._target_color_type == TargetType.CONTINUOUS:
            if(self.projection==3):
                self.fig.colorbar(self._scatter, ax=self.ax)
            
            else:
                self._layout()
                # Manually draw the colorbar.
                self.cbar = mpl.colorbar.ColorbarBase(self.cax, cmap=self._colors, 
                                                  norm=self._norm)

    def _determine_scatter_kwargs(self, y=None):
        """
        Determines scatter argumnets to pass into ``plt.scatter()``. If y is
        discrete or single then determine colors. If continuous then determine
        colors and colormap.Also normalize to range

        Parameters
        ----------
        y : array-like of shape (n,), optional
            The target, used to specify the colors of the points for continuous
            target.
        """

        scatter_kwargs = {"alpha": self.alpha}
        # Determine the colors
        if self._target_color_type == TargetType.SINGLE:
            scatter_kwargs["c"] = self._colors

        elif self._target_color_type == TargetType.DISCRETE:
            if y is None:
                raise YellowbrickValueError("y is required for discrete target")

            try:
                scatter_kwargs["c"] = [self._colors[self.classes_[yi]] for yi in y]
            except IndexError:
                raise YellowbrickValueError("Target needs to be label encoded.")


        elif self._target_color_type == TargetType.CONTINUOUS:
            if y is None:
                raise YellowbrickValueError("y is required for continuous target")

            scatter_kwargs["c"] = y
            scatter_kwargs["cmap"] = self._colors
            self._norm = mpl.colors.Normalize(vmin=self.range_[0], vmax=self.range_[1])

        else:
            # Technically this should never be raised
            raise NotFitted("could not determine target color type")
        return scatter_kwargs;