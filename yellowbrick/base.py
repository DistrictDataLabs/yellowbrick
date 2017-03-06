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

    def __init__(self, ax=None, **kwargs):
        """
        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.

        Parameters
        ----------
        ax: matplotlib axes
            the axis to plot the figure on.

        kwargs: dict
            keyword arguments passed to the super class.
        """
        self.ax = ax
        self.size  = kwargs.pop('size', None)
        self.color = kwargs.pop('color', None)
        self.title = kwargs.pop('title', None)

    def fit(self, X, y=None, **kwargs):
        """
        Fits a transformer to X and y

        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: dict
            keyword arguments passed to Scikit-Learn API.
        """
        return self

    def gca(self):
        """
        Creates axes if they don't already exist
        """
        if self.ax is None:
            self.ax = plt.gca()
        return self.ax

    def draw(self, **kwargs):
        """
        Rendering function
        """
        ax = self.gca()

    def poof(self, outpath=None, **kwargs):
        """
        The user calls poof, which is the primary entry point
        for producing a visualization.

        Visualizes either data features or fitted model scores

        Parameters
        ----------
        outpath: string
            path or None. Save  figure to disk or if None show in window

        kwargs: generic keyword arguments.
        """
        if self.ax is None: return

        self.finalize()

        if outpath is not None:
            plt.savefig(outpath, **kwargs)
        else:
            plt.show()

    def set_title(self, title=None):
        """
        Sets the title on the current axes.
        """
        title = self.title or title
        if title is not None:
            self.ax.set_title(title)

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.

        Parameters
        ----------
        kwargs: dict
            generic keyword arguments.
        """
        pass

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

    def __init__(self, model, ax=None, **kwargs):
        """
        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.

        Parameters
        ----------
        models: object
            the Scikit-Learn models being compared with each other.

        ax: matplotlib axes
            the axis to plot the figure on.

        kwargs: dict
            keyword arguments.
        """
        super(ScoreVisualizer, self).__init__(ax=ax, **kwargs)

        self.estimator = model
        self.name = get_model_name(self.estimator)

    def fit(self, X, y=None, **kwargs):
        """
        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: dict
            keyword arguments passed to Scikit-Learn API.
        """
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def draw(self, X, y):
        """
        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values
        """
        pass


##########################################################################
## Model Visualizers
##########################################################################

class ModelVisualizer(Visualizer):
    """
    A model visualization accepts as input an unfitted Scikit-Learn estimator(s)
    and enables the user to visualize the performance of models across a range
    of hyperparameter values (e.g. using VisualGridsearch and ValidationCurve).
    """
    def __init__(self, model, ax=None, **kwargs):
        """
        Parameters
        ----------
        ax: matplotlib axes
            the axis to plot the figure on.

        kwargs: dict
            keyword arguments for Scikit-Learn model

        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.
        """
        super(ModelVisualizer, self).__init__(ax=ax, **kwargs)
        self.estimator = model


    def fit(self, X, y=None, **kwargs):
        """
        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: dict
            keyword arguments passed to Scikit-Learn API.
        """
        pass

    def predict(self, X):
        pass

##########################################################################
## Multiple Models and Mixins
##########################################################################

class MultiModelMixin(object):
    """
    Does predict for each of the models and generates subplots.
    """

    def __init__(self, models, ax=None, **kwargs):
        # Ensure models is a collection, if it's a single estimator then we
        # wrap it in a list so that the API doesn't break during render.
        """
        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.

        Parameters
        ----------
        models: the Scikit-Learn models being compared with each other.

        kwargs: dict
            keyword arguments.
        """
        # TODO: How to handle the axes in this mixin?
        self.ax = ax

        if all(isestimator, models):
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
