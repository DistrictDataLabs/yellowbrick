"""
Colorplot visualizer for gridsearch results.
"""

import numpy as np

from .base import GridSearchVisualizer


## Packages for export
__all__ = [
    "GridSearchColorPlot",
    "gridsearch_color_plot"
]


##########################################################################
## Quick method
##########################################################################

def gridsearch_color_plot(model, x_param, y_param, X=None, y=None, ax=None,
                          **kwargs):
    """Quick method:
    Create a color plot showing the best grid search scores across two
    parameters.

    This helper function is a quick wrapper to utilize GridSearchColorPlot
    for one-off analysis.

    If no `X` data is passed, the model is assumed to be fit already. This
    allows quick exploration without waiting for the grid search to re-run.

    Parameters
    ----------
    model : Scikit-Learn grid search object
        Should be an instance of GridSearchCV. If not, an exception is raised.
        The model may be fit or unfit.

    x_param : string
        The name of the parameter to be visualized on the horizontal axis.

    y_param : string
        The name of the parameter to be visualized on the vertical axis.

    metric : string (default 'mean_test_score')
        The field from the grid search's `cv_results` that we want to display.

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
    visualizer = GridSearchColorPlot(model, x_param, y_param, ax=ax, **kwargs)

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

    x_param : string
        The name of the parameter to be visualized on the horizontal axis.

    y_param : string
        The name of the parameter to be visualized on the vertical axis.

    metric : string (default 'mean_test_score')
        The field from the grid search's `cv_results` that we want to display.

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
    >>> model = GridSearchColorPlot(gridsearch, x_param='kernel', y_param='C')
    >>> model.fit(X)
    >>> model.poof()
    """

    def __init__(self, model, x_param, y_param, metric='mean_test_score',
                 colormap='RdBu_r', ax=None, **kwargs):
        super(GridSearchColorPlot, self).__init__(model, ax=ax, **kwargs)
        self.x_param = x_param
        self.y_param = y_param
        self.metric = metric
        self.colormap = colormap

    def draw(self):
        # Project the grid search results to 2 dimensions
        x_vals, y_vals, best_scores = self.param_projection(
            self.x_param, self.y_param, metric=self.metric
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
        self.ax.set_xlabel(self.x_param)
        self.ax.set_ylabel(self.y_param)
