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
