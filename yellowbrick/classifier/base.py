# yellowbrick.classifier.base
# API for classification visualizer hierarchy.
#
# Author:   Rebecca Bilbro
# Author:   Benjamin Bengfort
# Author:   Neal Humphrey
# Created:  Wed May 18 12:39:40 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: base.py [5388065] neal@nhumphrey.com $

"""
API for classification visualizer hierarchy.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from yellowbrick.utils import isclassifier
from yellowbrick.base import ScoreVisualizer
from yellowbrick.style.palettes import color_palette
from yellowbrick.exceptions import YellowbrickTypeError


##########################################################################
## Base Classification Visualizer
##########################################################################


class ClassificationScoreVisualizer(ScoreVisualizer):
    def __init__(
        self, model, ax=None, fig=None, classes=None, is_fitted="auto", **kwargs
    ):
        """
        Check to see if model is an instance of a classifer.
        Should return an error if it isn't.

        Parameters
        -----------
        model : sklearn.Estimator (a classifier)
            ClassificationScoreVisualizer wraps a classifier to produce a
            visualization of its score. If the internal model is not fitted,
            it is fit when the visualizer is fitted, unless otherwise specified
            by ``is_fitted``.

        ax : matplotlib Axes, default: None
            The axis to plot the figure on. If None is passed in the current axes
            will be used (or generated if required).

        fig : matplotlib Figure, default: None
            The figure to plot the Visualizer on. If None is passed in the current
            plot will be used (or generated if required).

        classes : a list of class names for the legend
            If classes is None and a y value is passed to fit then the classes
            are selected from the target vector.

        is_fitted : bool or str, default="auto"
            Specify if the wrapped estimator is already fitted. If False, the estimator
            will be fit when the visualizer is fit, otherwise, the estimator will not be
            modified. If "auto" (default), a helper method will check if the estimator
            is fitted before fitting it again.

        kwargs : dict
            Keyword arguments that are passed to the base class and may influence
            the visualization as defined in other Visualizers. Optional keyword
            arguments include:

        .. todo:: Finish documenting class.
        .. todo:: accept as input ``classes``, as all visualizers need this.
        """
        # A bit of type checking
        if not isclassifier(model):
            raise YellowbrickTypeError(
                "This estimator is not a classifier; "
                "try a regression or clustering score visualizer instead!"
            )

        # Convert to array if necessary to match estimator.classes_
        if classes is not None:
            classes = np.array(classes)

        # Set up classifier score visualization properties
        if classes is not None:
            n_colors = len(classes)
        else:
            n_colors = None

        self.colors = color_palette(kwargs.pop("colors", None), n_colors)
        self.classes_ = classes

        # Initialize the super method.
        super(ClassificationScoreVisualizer, self).__init__(
            model, ax=ax, fig=fig, **kwargs
        )

    @property
    def classes_(self):
        """
        Proxy property to smartly access the classes from the estimator or
        stored locally on the score visualizer for visualization.
        """
        if self.__classes is None:
            try:
                return self.estimator.classes_
            except AttributeError:
                return None  # TODO: raise NotFittedError instead of returning None
        return self.__classes

    @classes_.setter
    def classes_(self, value):
        self.__classes = value

    def fit(self, X, y=None, **kwargs):
        """
        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: dict
            Keyword arguments passed to Scikit-Learn API.

        Returns
        -------
        self : instance
            Returns the instance of the classification score visualizer

        """
        super(ClassificationScoreVisualizer, self).fit(X, y, **kwargs)

        # Extract the classes from the estimator
        if self.classes_ is None:
            self.classes_ = self.estimator.classes_

        # Always return self from fit
        return self

    def score(self, X, y, **kwargs):
        """
        The score function is the hook for visual interaction. Pass in test
        data and the visualizer will create predictions on the data and
        evaluate them with respect to the test values. The evaluation will
        then be passed to draw() and the result of the estimator score will
        be returned.

        Parameters
        ----------
        X : array-like
            X (also X_test) are the dependent variables of test set to predict
        y : array-like
            y (also y_test) is the independent actual variables to score against

        Returns
        -------
        score : float
        """
        self.score_ = self.estimator.score(X, y, **kwargs)

        return self.score_

    # TODO during refactoring this can be used to generalize ClassBalance
    def class_counts(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))
