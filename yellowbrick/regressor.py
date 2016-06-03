# yellowbrick.regressor
# Visualizations related to evaluating Scikit-Learn regressor models
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Jun 03 10:30:36 2016 -0700
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: regressor.py [] benjamin@bengfort.com $

"""
Visualizations related to evaluating Scikit-Learn regressor models
"""

##########################################################################
## Imports
##########################################################################

import matplotlib as mpl
import matplotlib.pyplot as plt

from .base import ModelVisualization
from .utils import get_model_name, isestimator
from sklearn.cross_validation import cross_val_predict as cvp

##########################################################################
## Regression Visualization Base Object
##########################################################################

class RegressorVisualization(ModelVisualization):
    pass


##########################################################################
## Prediction Error Plots
##########################################################################

class PredictionError(RegressorVisualization):

    def __init__(self, models, **kwargs):
        """
        Pass in a collection of models to generate prediction error graphs.
        """

        # Ensure models is a collection, if it's a single estimator then we
        # wrap it in a list so that the API doesn't break during render.
        if isestimator(models):
            models = [models]

        # Keep track of the models
        self.models = models
        self.names  = kwargs.pop('names', list(map(get_model_name, models)))
        self.colors = {
            'point': kwargs.pop('point_color', '#F2BE2C'),
            'line': kwargs.pop('line_color', '#2B94E9'),
        }

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

    def render(self, X, y):
        """
        Renders each of the scatter plots per matrix.
        """
        for idx, (axe, y_pred) in enumerate(zip(self.generate_subplots(), self.predict(X, y))):
            # Plot the correct values
            axe.scatter(y, y_pred, c=self.colors['point'])

            # Draw the best fit line
            # TODO: Add best fit line computation metric
            axe.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4, c=self.colors['line'])

            # Set the title and the y-axis label
            axe.set_title("Predicted vs. Actual Values for {}".format(self.names[idx]))
            axe.set_ylabel('Predicted Value')

        # Finalize figure
        plt.xlabel('Actual Value')
        return axe # TODO: We shouldn't return the last axis


def peplot(models, X, y, **kwargs):
    viz = PredictionError(models, **kwargs)
    return viz.render(X, y)
