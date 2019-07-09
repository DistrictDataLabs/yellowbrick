import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


import matplotlib as mpl

from yellowbrick.draw import manual_legend
from yellowbrick.features.base import *
from yellowbrick.exceptions import YellowbrickValueError, NotFitted


##########################################################################
## Projection Visualizers
##########################################################################

class ProjectionVisualizer(DataVisualizer):
    
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
        Creates the layout for colorbar. The colorbar is added to the right of the 
        scatterplot 
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
        
        super(ProjectionVisualizer, self).fit(X, y, **kwargs)
        self.transformer.fit(X)
        return self

    def transform(self, X, y=None, **kwargs):
        
        try:
            Xp = self.transformer.transform(X)
        except AttributeError as e:
            raise AttributeError(str(e) + " try using fit_transform instead.")
        self.draw(Xp, y, **kwargs)
        return Xp
    
    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y, **kwargs).transform(X, y, **kwargs)
    
    def draw(self, X, y=None):

        scatter_kwargs = self._determine_scatter_kwargs(y);
        
        if self.projection == 2:
            self._scatter = self.ax.scatter(X[:,0], X[:,1], **scatter_kwargs)

        if self.projection == 3:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self._scatter = self.ax.scatter(X[:, 0], X[:, 1], X[:, 2], **scatter_kwargs)
        
        return self.ax

    def finalize(self):
        
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