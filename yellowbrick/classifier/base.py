# yellowbrick.classifier.base
# API for classification visualizer hierarchy.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Author:   Neal Humphrey
# Created:  Wed May 18 12:39:40 2016 -0400
#
# Copyright (C) 2016 District Data Labs
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

from ..utils import isclassifier
from ..base import ScoreVisualizer
from ..style.palettes import color_palette
from ..exceptions import YellowbrickTypeError


##########################################################################
## Base Classification Visualizer
##########################################################################

class ClassificationScoreVisualizer(ScoreVisualizer):

    def __init__(self, model, ax=None, classes=None, **kwargs):
        """
        Check to see if model is an instance of a classifer.
        Should return an error if it isn't.

        .. todo:: document this class.
        .. tood:: accept as input classes as all visualizers need this.
        """
        # A bit of type checking
        if not isclassifier(model):
            raise YellowbrickTypeError(
                "This estimator is not a classifier; "
                "try a regression or clustering score visualizer instead!"
        )

        # Initialize the super method.
        super(ClassificationScoreVisualizer, self).__init__(model, ax=ax, **kwargs)

        # Convert to array if necessary to match estimator.classes_
        if classes is not None:
            classes = np.array(classes)

        # Set up classifier score visualization properties
        if classes is not None:
            n_colors = len(classes)
        else:
            n_colors = None

        self.colors    = color_palette(kwargs.pop('colors', None), n_colors)
        self.classes_  = classes

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
                return None
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

        kwargs: keyword arguments passed to Scikit-Learn API.

        Returns
        -------
        self : instance
            Returns the instance of the classification score visualizer

        """
        # Fit the inner estimator
        self.estimator.fit(X, y)

        # Extract the classes from the estimator
        if self.classes_ is None:
            self.classes_ = self.estimator.classes_

        # Always return self from fit
        return self

    #TODO during refactoring this can be used to generalize ClassBalance
    def class_counts(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))
