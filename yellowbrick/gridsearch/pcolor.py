"""
Colorplot visualizer for gridsearch results.
"""

import time
import numpy as np

from .base import GridSearchVisualizer
from ..exceptions import YellowbrickValueError

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from matplotlib.mlab import griddata


## Packages for export
__all__ = [
    "GridSearchColorPlot",
    "gridsearch_color_plot"
]


##########################################################################
## Dimension reduction utility
##########################################################################

def param_projection(cv_results, param_1, param_2):
    """
    Projects the grid search results onto 2 dimensions.

    The display value is taken as the max over the non-displayed dimensions.

    Parameters
    ----------
    cv_results : dict
        A dictionary of results from the `GridSearchCV` object's `cv_results_`
        attribute

    param_1 : string
        The name of the parameter to be visualized on the horizontal axis.

    param_2 : string
        The name of the parameter to be visualized on the vertical axis.

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the classification report was drawn on.
    """
    # Get unique values of the two display parameters
    x_vals = sorted(list(set(cv_results['param_' + param_1].compressed())))
    y_vals = sorted(list(set(cv_results['param_' + param_2].compressed())))
    n_x = len(x_vals)
    n_y = len(y_vals)

    # Get mapping from parameter value -> integer index
    int_mapping_1 = {value: idx for idx, value in enumerate(x_vals)}
    int_mapping_2 = {value: idx for idx, value in enumerate(y_vals)}

    # Translate each gridsearch result to indices on the grid
    idx_x = [int_mapping_1[value] if value else None
             for value in cv_results['param_' + param_1]]
    idx_y = [int_mapping_2[value] if value else None
             for value in cv_results['param_' + param_2]]

    # Create an array of all scores for each value of the display parameters.
    # This is a n_x by n_y array of lists with `None` in place of empties
    # (my kingdom for a dataframe...)
    all_scores = [[None for _ in range(n_x)] for _ in range(n_y)]
    for x, y, score in zip(idx_x, idx_y, cv_results['mean_test_score']):
        if x is not None and y is not None:
            if all_scores[y][x] is None:
                all_scores[y][x] = []
            all_scores[y][x].append(score)

    # Get a numpy array consisting of the best scores for each parameter pair
    best_scores = np.empty((n_y, n_x))
    for x in range(n_x):
        for y in range(n_y):
            if all_scores[y][x] is None:
                best_scores[y, x] = np.nan
            else:
                best_scores[y, x] = max(all_scores[y][x])

    return x_vals, y_vals, best_scores


##########################################################################
## Quick method
##########################################################################

def gridsearch_color_plot(model, param_1, param_2, X=None, y=None, ax=None,
                          **kwargs):
    """Quick method:
    Create a color plot showing the best grid search scores across two
    parameters.

    This helper function is a quick wrapper to utilize GridSearchColorPlot
    for one-off analysis.

    If no `X` data is passed, the model is assume to be fit already. This
    allows quick exploration without waiting for the grid search to re-run.

    Parameters
    ----------
    model : Scikit-Learn grid search object
        Should be an instance of GridSearchCV. If not, an exception is raised.
        The model may be fit or unfit.

    param_1 : string
        The name of the parameter to be visualized on the horizontal axis.

    param_2 : string
        The name of the parameter to be visualized on the vertical axis.

    X  : ndarray or DataFrame of shape n x m or None (default None)
        A matrix of n instances with m features. If not None, forces the
        GridSearchCV object to be fit.

    y  : ndarray or Series of length n or None (default None)
        An array or series of target or class values.

    ax : matplotlib axes
        The axes to plot the figure on.

    classes : list of strings
        The names of the classes in the target

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the classification report was drawn on.
    """
    # Instantiate the visualizer
    visualizer = GridSearchColorPlot(model, param_1, param_2, ax=ax, **kwargs)

    # Fit if necessary
    if X is not None:
        visualizer.fit(X, y)
    else:
        visualizer.draw()

    # Return the axes object on the visualizer
    return visualizer.ax


class GridSearchColorPlot(GridSearchVisualizer):
    """
    Create a color plot showing the best grid search scores across two
    parameters.

    Parameters
    ----------
    model : Scikit-Learn grid search object
        Should be an instance of GridSearchCV. If not, an exception is raised.

    param_1 : string
        The name of the parameter to be visualized on the horizontal axis.

    param_2 : string
        The name of the parameter to be visualized on the vertical axis.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    colormap : string or cmap, default: 'RdBu_r'
        optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or
        colormap to color them on a continuous scale.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------
    >>> from yellowbrick.gridsearch import GridSearchColorPlot
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import SVC
    >>> gridsearch = GridSearchCV(SVC(),
                                  {'kernel': ['rbf', 'linear'], 'C': [1, 10]})
    >>> model = GridSearchColorPlot(gridsearch, param_1='kernel', param_2='C')
    >>> model.fit(X)
    >>> model.poof()
    """

    def __init__(self, model, param_1, param_2, colormap='RdBu_r', ax=None,
                 **kwargs):
        # super(GridSearchColorPlot, self).__init__(model, ax=ax, **kwargs)
        super().__init__(model, ax=ax, **kwargs)
        self.param_1 = param_1
        self.param_2 = param_2
        self.colormap = colormap

    def draw(self):
        # Project the grid search results to 2 dimensions
        x_vals, y_vals, best_scores = param_projection(
            self.cv_results_, self.param_1, self.param_2
        )

        # Mask nans so that they can be filled with a hatch
        data = np.ma.masked_invalid(best_scores)

        # Plot and fill in hatch for nans
        mesh = self.ax.pcolor(data, cmap=self.colormap,
                              vmin=np.nanmin(data), vmax=np.nanmax(data))
        self.ax.patch.set(hatch='x', edgecolor='black')

        # Ticks and tick labels
        self.ax.set_xticks(np.arange(len(x_vals)) + 0.5)
        self.ax.set_yticks(np.arange(len(y_vals)) + 0.5)
        self.ax.set_xticklabels(x_vals, rotation=45)
        self.ax.set_yticklabels(y_vals, rotation=45)

        # Add the colorbar
        cb = self.ax.figure.colorbar(mesh, None, self.ax)
        cb.outline.set_linewidth(0)

        self.ax.set_aspect("equal")

    def finalize(self):
        self.set_title("Grid Search Scores")
        self.ax.set_xlabel(self.param_1)
        self.ax.set_ylabel(self.param_2)
