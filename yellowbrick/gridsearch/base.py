"""
Base class for grid search visualizers
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
from ..utils import is_gridsearch
from ..base import ModelVisualizer
from ..exceptions import (YellowbrickTypeError,
                          YellowbrickKeyError,
                          YellowbrickValueError)


##########################################################################
## Dimension reduction utility
##########################################################################

def param_projection(cv_results, x_param, y_param, metric='mean_test_score'):
    """
    Projects the grid search results onto 2 dimensions.

    The display value is taken as the max over the non-displayed dimensions.

    Parameters
    ----------
    cv_results : dict
        A dictionary of results from the `GridSearchCV` object's `cv_results_`
        attribute.

    x_param : string
        The name of the parameter to be visualized on the horizontal axis.

    y_param : string
        The name of the parameter to be visualized on the vertical axis.

    metric : string (default 'mean_test_score')
        The field from the grid search's `cv_results` that we want to display.

    Returns
    -------
    unique_x_vals : list
        The parameter values that will be used to label the x axis.

    unique_y_vals: list
        The parameter values that will be used to label the y axis.

    best_scores: 2D numpy array (n_y by n_x)
        Array of scores to be displayed for each parameter value pair.
    """
    # Extract the parameter values and score corresponding to each gridsearch
    # trial.
    # These are masked arrays where the cases where each parameter is
    # non-applicable are masked.
    try:
        x_vals = cv_results['param_' + x_param]
    except KeyError:
        raise YellowbrickKeyError("Parameter '{}' does not exist in the grid "
                                  "search results".format(x_param))
    try:
        y_vals = cv_results['param_' + y_param]
    except KeyError:
        raise YellowbrickKeyError("Parameter '{}' does not exist in the grid "
                                  "search results".format(y_param))

    if metric not in cv_results:
        raise YellowbrickKeyError("Metric '{}' does not exist in the grid "
                                  "search results".format(metric))

    # Get unique, unmasked values of the two display parameters
    unique_x_vals = sorted(list(set(x_vals.compressed())))
    unique_y_vals = sorted(list(set(y_vals.compressed())))
    n_x = len(unique_x_vals)
    n_y = len(unique_y_vals)

    # Get mapping of each parameter value -> an integer index
    int_mapping_1 = {value: idx for idx, value in enumerate(unique_x_vals)}
    int_mapping_2 = {value: idx for idx, value in enumerate(unique_y_vals)}

    # Translate each gridsearch result to indices on the grid
    idx_x = [int_mapping_1[value] if value else None for value in x_vals]
    idx_y = [int_mapping_2[value] if value else None for value in y_vals]

    # Create an array of all scores for each value of the display parameters.
    # This is a n_x by n_y array of lists with `None` in place of empties
    # (my kingdom for a dataframe...)
    all_scores = [[None for _ in range(n_x)] for _ in range(n_y)]
    for x, y, score in zip(idx_x, idx_y, cv_results[metric]):
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
                try:
                    best_scores[y, x] = max(all_scores[y][x])
                except ValueError:
                    raise YellowbrickValueError(
                        "Cannot display grid search results for metric '{}': "
                        "result values may not all be numeric".format(metric)
                    )

    return unique_x_vals, unique_y_vals, best_scores


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

    def param_projection(self, x_param, y_param, metric):
        """
        Projects the grid search results onto 2 dimensions.

        The wrapped GridSearch object is assumed to be fit already.
        The display value is taken as the max over the non-displayed dimensions.

        Parameters
        ----------
        x_param : string
            The name of the parameter to be visualized on the horizontal axis.

        y_param : string
            The name of the parameter to be visualized on the vertical axis.

        metric : string (default 'mean_test_score')
            The field from the grid search's `cv_results` that we want to display.

        Returns
        -------
        unique_x_vals : list
            The parameter values that will be used to label the x axis.

        unique_y_vals: list
            The parameter values that will be used to label the y axis.

        best_scores: 2D numpy array (n_y by n_x)
            Array of scores to be displayed for each parameter value pair.
        """
        return param_projection(self.estimator.cv_results_, x_param, y_param, metric)

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
