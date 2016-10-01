# yellowbrick.regressor
# Visualizations related to evaluating Scikit-Learn regressor models
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Jun 03 10:30:36 2016 -0700
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: regressor.py [4a59c49] benjamin@bengfort.com $

"""
Visualizations related to evaluating Scikit-Learn regressor models
"""

##########################################################################
## Imports
##########################################################################

import matplotlib as mpl
import matplotlib.pyplot as plt

from .bestfit import draw_best_fit
from .utils import get_model_name, isestimator
from .base import Visualizer, ScoreVisualizer, MultiModelMixin
from sklearn.cross_validation import train_test_split as tts

##########################################################################
## Regression Visualization Base Object
##########################################################################

class RegressorVisualization(ScoreVisualizer):
    pass


##########################################################################
## Prediction Error Plots
##########################################################################

class PredictionError(MultiModelMixin, RegressorVisualization):

    def __init__(self, models, **kwargs):
        """
        Pass in a collection of models to generate prediction error graphs.
        """
        super(PredictionError, self).__init__(models, **kwargs)

        self.colors = {
            'point': kwargs.pop('point_color', '#F2BE2C'),
            'line': kwargs.pop('line_color', '#2B94E9'),
        }

    def render(self, X, y):
        """
        Renders each of the scatter plots per matrix.
        """
        for idx, (axe, y_pred) in enumerate(zip(self.generate_subplots(), self.predict(X, y))):
            # Set the x and y limits
            axe.set_xlim(y.min()-1, y.max()+1)
            axe.set_ylim(y_pred.min()-1, y_pred.max()+1)

            # Plot the correct values
            axe.scatter(y, y_pred, c=self.colors['point'])

            # Draw the linear best fit line on the regression
            draw_best_fit(y, y_pred, axe, 'linear', ls='--', lw=2, c=self.colors['line'])

            # Set the title and the y-axis label
            axe.set_title("Predicted vs. Actual Values for {}".format(self.names[idx]))
            axe.set_ylabel('Predicted Value')

        # Finalize figure
        plt.xlabel('Actual Value')
        return axe # TODO: We shouldn't return the last axis


def peplot(models, X, y, **kwargs):
    """
    Take in the model, data and labels as input and generate a multi-plot of
    the prediction error for each model.
    """
    viz = PredictionError(models, **kwargs)
    return viz.render(X, y)

##########################################################################
## Residuals Plots
##########################################################################

class ResidualsPlot(MultiModelMixin, RegressorVisualization):
    """
    Unlike PredictionError, this viz takes classes instead of model instances
    we should revise the API to have FittedRegressorVisualization vs. etc.

    TODO: Fitted vs. Unfitted API.
    """

    def __init__(self, models, **kwargs):
        """
        Pass in a collection of model classes to generate train/test residual
        plots by fitting the models and ... someone finish this docstring.
        """
        super(ResidualsPlot, self).__init__(models, **kwargs)

        # TODO: the names for the color arguments are _long_.
        self.colors = {
            'train_point': kwargs.pop('train_point_color', '#2B94E9'),
            'test_point': kwargs.pop('test_point_color', '#94BA65'),
            'line': kwargs.pop('line_color', '#333333'),
        }

    def fit(self, X, y):
        """
        Fit all three models and also store the train/test splits.

        TODO: move to MultiModelMixin.
        """
        # TODO: make test size a parameter and do better data storage on viz.
        self.X_train, self.X_test, self.y_train, self.y_test = tts(X, y, test_size=0.2)
        self.models = list(map(lambda model: model.fit(self.X_train, self.y_train), self.models))

    def render(self):
        """
        Renders each residual plot across each axis.
        """

        for idx, axe in enumerate(self.generate_subplots()):
            # Get the information for this axis
            model = self.models[idx]
            name  = self.names[idx]

            # TODO: less proceedural?
            # Add the training residuals
            y_train_pred = model.predict(self.X_train)
            axe.scatter(y_train_pred, y_train_pred - self.y_train, c=self.colors['train_point'], s=40, alpha=0.5)

            # Add the test residuals
            y_test_pred = model.predict(self.X_test)
            axe.scatter(y_test_pred, y_test_pred - self.y_test, c=self.colors['test_point'], s=40)

            # Add the hline and other axis elements
            # TODO: better parameters based on the plot or, normalize, then push -1 to 1
            axe.hlines(y=0, xmin=0, xmax=100)
            axe.set_title(name)
            axe.set_ylabel('Residuals')

        # Finalize the residuals plot
        # TODO: adjust the x and y ranges in order to compare (or use normalize)
        plt.xlabel("Predicted Value")
        return axe  # TODO: We shouldn't return the last axis


def residuals_plot(models, X, y, **kwargs):
    """
    Take in the model, data and labels as input and generate a multi-plot of
    the residuals for each.
    """
    viz = ResidualsPlot(models, **kwargs)
    viz.fit(X, y)
    return viz.render()
