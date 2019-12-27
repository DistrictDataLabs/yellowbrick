import numpy as np
import pandas as pd
from .base import RegressionScoreVisualizer
from yellowbrick.style.colors import resolve_colors
from yellowbrick.features.base import DataVisualizer
from yellowbrick.utils.target import TargetType


import matplotlib.pyplot as plt

##########################################################################
## Effect Plots
##########################################################################

class EffectPlot(RegressionScoreVisualizer):
    """
    Parameters
    ----------
  
    model : a Scikit-Learn regressor
        Should be an instance of a regressor, otherwise will raise a
        YellowbrickTypeError exception on instantiation.
        If the estimator is not fitted, it is fit when the visualizer is fitted,
        unless otherwise specified by ``is_fitted``.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes.
        will be used (or generated if required).

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
        
    marker : string, default:'D'
        Shape of the outliers in the boxplot. 'D' for diamond, '.' for 
        circles

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.
    """
    
    def __init__(self,model,ax=None, colors=None, colormap=None, marker='D', **kwargs):
        super(EffectPlot, self).__init__(model=model, ax=ax, **kwargs)
        self.colors = colors
        self.colormap = colormap
        self.marker=marker
        self.target_type = 'auto'
    
    def fit(self, X, y, **kwargs):
        """
        Fits the estimator to dataset and then multiply weights of a feature
        with each feature values to generate new dataset.
        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target values

        kwargs: keyword arguments passed to Scikit-Learn API.

        Returns
        -------
        self : EffectPlot
            The visualizer instance
        """
        categorical = self._get_categorical_columns(X)

        #Encoding
        dummy_df = pd.get_dummies(X, columns=categorical, drop_first=True)

        self.estimator.fit(dummy_df, y)
        coeffs = self.estimator.coef_
        dummy_df = dummy_df*coeffs
        
        # Recollect the columns
        for cols in categorical:
            col_names = [a for a in dummy_df.columns.values if cols in a]
            dummy_df[cols] = (dummy_df[col_names]).sum(axis=1)
            dummy_df.drop(columns = col_names, inplace=True)
              
        self.draw(dummy_df)
        self.columns = dummy_df.columns 
        return self
        
    def _get_categorical_columns(self, X):
        """
        Finds the categorical features among the given features. Uses
        _determine_targer_color_type from `DataVisualizer` for this purpose.
        """
        categorical=[]
        cols = X.columns
        for col in cols:
            DataVisualizer._determine_target_color_type(self, X[col])
            if(self._target_color_type==TargetType.DISCRETE):
                categorical.append(col)
        return categorical
        
        
    def draw(self, X, **kwargs):
        """
        Draws a box plot. Provide control over almost anything in the plot.
        """
        colors = resolve_colors(
                    n_colors=len(X.columns.values), 
                    colormap=self.colormap, 
                    colors=self.colors
                    )
        self.ax.vlines(x=0,ymin=0,ymax=X.shape[1]+1,linestyles='dashed', alpha=0.5)
        X = np.array(X)
        bp = self.ax.boxplot(X, sym='black', patch_artist=True,vert=False)
        for (box,color) in zip(bp['boxes'],colors):
            box.set( facecolor=color, linewidth=2)
        for median in bp['medians']:
            median.set(color='black', linewidth=3)
        for whisker in bp['whiskers']:
            whisker.set(color='black', linewidth=2)
        for cap in bp['caps']:
            cap.set(color='black', linewidth=2)
        for flier in bp['fliers']:
            # TODO: Set color of flier here
            flier.set(marker=self.marker)
        return self.ax

        
    def finalize(self):
        self.ax.set_xlabel('Feature effect')
        self.ax.set_yticklabels(self.columns)
        self.ax.set_title('Effect Plot')
        plt.tight_layout()
        
        
##########################################################################
## Quick Method
##########################################################################
        
def effectplot(model, X, y, ax=None, colors=None, colormap=None, marker='D', 
               show=True, **kwargs):
    """
    Quick Method.
    """
    viz = EffectPlot(model=model, ax=ax, colors=colors, colormap=colormap, 
                            marker=marker, **kwargs)
    viz.fit(X, y)
       
    if show:
        viz.show()
    else:
        viz.finalize()
    
    return viz

#Alias for Effectplot
EffectViz = EffectPlot