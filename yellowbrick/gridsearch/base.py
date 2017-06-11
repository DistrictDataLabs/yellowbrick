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
## Utility
##########################################################################

def param_projection(cv_results, display_params):
    # In which grid search results are the display_params present?
    params_present = np.array([True] * len(cv_results['mean_test_score']))
    for param in display_params:
        params_present *= ~cv_results['param_' + param].mask

    # Of these, which has the best cross-validation score?
    best_idx = np.argmax(cv_results['mean_test_score'][params_present])

    # Choose the best values of the non-display parameters and get a filter
    # narrowing the results to these best values
    other_params = {}
    filt = params_present
    for key in cv_results:
        # Look for parameters that are not display parameters...
        if key[:6]=='param_' and key[6:] not in display_params:
            # Get best vaue for this parameter
            best_value = cv_results[key][params_present][best_idx]
            other_params[key[6:]] = best_value

            # Filter results to this parameter value
            filt *= cv_results[key] == best_value
    print(other_params)

    return {k: np.array(cv_results[k])[filt] for k in cv_results}


#     # Get the unique values of each display parameter
#     param_values = [np.array(list(set(cv_results['param_' + param][filt])))
#                     for param in display_params]
#     param_lengths = [len(v) for v in param_values]
#     print(param_values)
#     print(param_lengths)
# }
#
#     # start with an array of nans, and fill in
#     mn_projection = np.ones(param_lengths) * np.nan
#     std_projection = np.ones(param_lengths) * np.nan
#     print(cv_results['mean_test_score'])
#     print(filt)
#     for idx in range(len(cv_results['mean_test_score'][filt])):
#         print(idx)
#         for param, values in zip(display_params, param_values):
#             print(param, values)
#             print(cv_results['param_' + param][idx])
#             print(cv_results['param_' + param][idx] == values)
#         array_idx = np.meshgrid([cv_results['param_' + param][idx] == values
#                                  for param, values in zip(display_params, param_values)])
#         print(array_idx)
#         print(type(array_idx))
#         print(array_idx.shape)
#         mn_projection[array_idx] = cv_results['mean_test_score']
#         std_projection[array_idx] = cv_results['std_test_score']
#
#     return mn_projection, std_projection




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
