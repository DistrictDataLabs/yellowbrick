"""
Colorplot visualizer for gridsearch results.
"""

import time
import numpy as np

from .base import GridSearchVisualizer, param_projection
from ..exceptions import YellowbrickValueError

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from matplotlib.mlab import griddata


## Packages for export
__all__ = [
    "GridSearchColorPlot",
]


class GridSearchColorPlot(GridSearchVisualizer):
    """
    TODO: description

    Parameters
    ----------

    model : Scikit-Learn grid search object
        Should be an instance of GridSearchCV. If not, an exception is raised.

    param_1 : string
        The name of the first parameter to be visualized

    param_2 : string
        The name of the second parameter to be visualized

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

    >>> from yellowbrick.cluster import KElbowVisualizer
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import SVC
    >>> gridsearch = GridSearchCV(SVC(), {'kernel': ['rbf', 'linear'], 'C': [1, 10]})
    >>> model = KElbowVisualizer(gridsearch, param_1='kernel', param_2='C')
    >>> model.fit(X)
    >>> model.poof()

    Notes
    -----


    .. todo::
    """

    def __init__(self, model, param_1, param_2, colormap='RdBu_r', ax=None,
                 **kwargs):
        super(GridSearchColorPlot, self).__init__(model, ax=ax, **kwargs)
        self.param_1 = param_1
        self.param_2 = param_2
        self.colormap = colormap

    def draw(self):

        cv_result = param_projection(self.cv_results_, (self.param_1, self.param_2))

        x_vals = sorted(list(set(cv_result['param_' + self.param_1])))
        y_vals = sorted(list(set(cv_result['param_' + self.param_2])))

        # get mapping from parameter value -> integer index
        int_mapping_1 = {value: idx for idx, value in enumerate(x_vals)}
        int_mapping_2 = {value: idx for idx, value in enumerate(y_vals)}

        # translate each gridsearch result to indices
        idx_x = [int_mapping_1[value] for value in cv_result['param_' + self.param_1]]
        idx_y = [int_mapping_2[value] for value in cv_result['param_' + self.param_2]]

        n_x = max(idx_x) + 1
        n_y = max(idx_y) + 1

        z = np.ones((n_y, n_x)) * np.nan
        for x, y, score in zip(idx_x, idx_y, cv_result['mean_test_score']):
            z[y, x] = score

        mesh = self.ax.pcolor(z, cmap=self.colormap)
        self.ax.set_xticks(np.arange(n_x) + 0.5)
        self.ax.set_yticks(np.arange(n_y) + 0.5)

        self.ax.set_xticklabels(x_vals, rotation=45)
        self.ax.set_yticklabels(y_vals, rotation=45)

        # TODO: strict proportion
        self.ax.set_aspect("equal")

        # Add the colorbar
        cb = self.ax.figure.colorbar(mesh, None, self.ax)
        cb.outline.set_linewidth(0)

    def finalize(self):
        self.ax.set_xlabel(self.param_1)
        self.ax.set_ylabel(self.param_2)
