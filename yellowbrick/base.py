# yellowbrick.base
# Abstract base classes and interface for Yellowbrick.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Jun 03 10:20:59 2016 -0700
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: base.py [4a59c49] benjamin@bengfort.com $

"""
Abstract base classes and interface for Yellowbrick.
"""

import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from .exceptions import YellowbrickTypeError
from .utils import get_model_name, isestimator
from sklearn.cross_validation import cross_val_predict as cvp


##########################################################################
## Base class hierarchy
##########################################################################

class Visualizer(BaseEstimator):
    """
    The root of the visual object hierarchy that defines how yellowbrick
    creates, stores, and renders visual artifacts using matplotlib.

    Inherits from Scikit-Learn's BaseEstimator class.

    The base class for feature visualization and model visualization
    primarily ensures that styling arguments are passed in.
    """

    def __init__(self, **kwargs):
        self.size  = kwargs.pop('size', None)
        self.color = kwargs.pop('color', None)

    def fit(self, X, y=None, **kwargs):
        """
        Fits a transformer to X and y
        """
        return self

    def draw(self, **kwargs):
        pass

    def poof(self, **kwargs):
        """
        The user calls poof, which is the primary entry point
        for producing a visualization.

        Visualizes either data features or fitted model scores
        """
        raise NotImplementedError(
            "All visualizations must specify their own poof methodology"
        )

    def fit_draw(self, X, y=None, **kwargs):
        """
        Fits a transformer to X and y then returns
        visualization of features or fitted model.
        """
        self.fit(X, y, **kwargs)
        self.draw(**kwargs)

    def fit_draw_poof(self, X, y=None, **kwargs):
        self.fit_draw(X, y, **kwargs)
        self.poof(**kwargs)


##########################################################################
## Score Visualizers
##########################################################################

class ScoreVisualizer(Visualizer):
    """
    Base class to follow an estimator in a visual pipeline.

    Draws the score for the fitted model.
    """

    def __init__(self, model, **kwargs):
        self.estimator = model
        super(ScoreVisualizer, self).__init__(**kwargs)

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def score(self, X, y=None):
        """
        Score will call draw to visualize model performance.
        If y_pred is None, call fit-predict on the model to get a y_pred.

        Score calls draw
        """
        y_pred = self.predict(X)
        return self.draw(y,y_pred)

    def draw(self, X, y):
        pass

    def poof(self, **kwargs):
        """
        The user calls poof
        """
        raise NotImplementedError(
            "Please specify how to render the feature visualization"
        )


##########################################################################
## Model Visualizers
##########################################################################

class ModelVisualizer(Visualizer):
    """
    A model visualization accepts as input an unfitted Scikit-Learn estimator(s)
    and enables the user to visualize the performance of models across a range
    of hyperparameter values (e.g. using VisualGridsearch and ValidationCurve).
    """

    def fit(self, X, y=None, **kwargs):
        pass

    def predict(self, X):
        pass

    def poof(self, model=None):
        """
        The user calls poof.

        A model visualization renders a model
        """
        raise NotImplementedError(
            "Please specify how to render the model visualization"
        )


##########################################################################
## Multiple Models and Mixins
##########################################################################

class MultiModelMixin(object):
    """
    Does predict for each of the models and generates subplots.
    """

    def __init__(self, models, **kwargs):
        # Ensure models is a collection, if it's a single estimator then we
        # wrap it in a list so that the API doesn't break during render.
        if isestimator(models):
            models = [models]

        # Keep track of the models
        self.models = models
        self.names  = kwargs.pop('names', list(map(get_model_name, models)))

    def generate_subplots(self):
        """
        Generates the subplots for the number of given models.
        """
        _, axes = plt.subplots(len(self.models), sharex=True, sharey=True)
        return axes

    def predict(self, X, y):
        """
        Returns a generator containing the predictions for each of the
        internal models (using cross_val_predict and a CV=12).
        """
        for model in self.models:
            yield cvp(model, X, y, cv=12)
