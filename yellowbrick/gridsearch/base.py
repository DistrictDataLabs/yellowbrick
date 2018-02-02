"""
Base class for grid search visualizers
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
from ..utils import is_gridsearch
from ..base import ModelVisualizer
from ..exceptions import YellowbrickTypeError


##########################################################################
## Dimension reduction utility
##########################################################################

def param_projection(cv_results, x_param, y_param):
    """
    Projects the grid search results onto 2 dimensions.

    The display value is taken as the max over the non-displayed dimensions.

    Parameters
    ----------
    cv_results : dict
        A dictionary of results from the `GridSearchCV` object's `cv_results_`
        attribute

    x_param : string
        The name of the parameter to be visualized on the horizontal axis.

    y_param : string
        The name of the parameter to be visualized on the vertical axis.

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the classification report was drawn on.
    """
    # Get unique values of the two display parameters
    x_vals = sorted(list(set(cv_results['param_' + x_param].compressed())))
    y_vals = sorted(list(set(cv_results['param_' + y_param].compressed())))
    n_x = len(x_vals)
    n_y = len(y_vals)

    # Get mapping from parameter value -> integer index
    int_mapping_1 = {value: idx for idx, value in enumerate(x_vals)}
    int_mapping_2 = {value: idx for idx, value in enumerate(y_vals)}

    # Translate each gridsearch result to indices on the grid
    idx_x = [int_mapping_1[value] if value else None
             for value in cv_results['param_' + x_param]]
    idx_y = [int_mapping_2[value] if value else None
             for value in cv_results['param_' + y_param]]

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
## Base Grid Search Visualizer
##########################################################################

class GridSearchVisualizer(ModelVisualizer):

    def __init__(self, model, ax=None, **kwargs):
        """
        Check to see if model is an instance of GridSearchCV.
        Should return an error if it isn't.
        """
        # A bit of type checking
        if not is_gridsearch(model):
            raise YellowbrickTypeError(
                "This estimator is not a GridSearchCV instance"
        )

        # Initialize the super method.
        super(GridSearchVisualizer, self).__init__(model, ax=ax, **kwargs)

    def param_projection(self, x_param, y_param):
        """
        Projects the grid search results onto 2 dimensions.

        The wrapped GridSearch object is assumed to be fit already.
        The display value is taken as the max over the non-displayed dimensions.
        """
        return param_projection(self.estimator.cv_results_, x_param, y_param)

    def fit(self, X, y=None, **kwargs):
        """
        Fits the wrapped grid search and calls draw().

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: dict
            Keyword arguments passed to the drawing functionality or to the
            Scikit-Learn API. See visualizer specific details for how to use
            the kwargs to modify the visualization or fitting process.

        Returns
        -------
        self : visualizer
            The fit method must always return self to support pipelines.
        """
        self.estimator.fit(X, y)
        self.draw()
        return self
