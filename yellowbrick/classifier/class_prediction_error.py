# yellowbrick.classifier.class_prediction_error
# Shows the balance of classes and their associated predictions.
#
# Author:   Larry Gray
# Author:   Benjamin Bengfort
# Created:  Fri Jul 20 10:26:25 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: class_prediction_error.py [] lwgray@gmail.com $

"""
Shows the balance of classes and their associated predictions.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.classification import _check_targets
from sklearn.model_selection import train_test_split as tts

from yellowbrick.draw import bar_stack
from yellowbrick.exceptions import ModelError, YellowbrickValueError
from yellowbrick.classifier.base import ClassificationScoreVisualizer


##########################################################################
## Class Prediction Error Chart
##########################################################################


class ClassPredictionError(ClassificationScoreVisualizer):
    """
    Class Prediction Error chart that shows the support for each class in the
    fitted classification model displayed as a stacked bar. Each bar is segmented
    to show the distribution of predicted classes for each class. It is initialized
    with a fitted model and generates a class prediction error chart on draw.

    Parameters
    ----------
    model: estimator
        Scikit-Learn estimator object. Should be an instance of a classifier,
        else ``__init__()`` will raise an exception. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.

    ax: axes, default: None
        the axis to plot the figure on.

    classes: list, default: None
        A list of class names for the legend. If classes is None and a y value
        is passed to fit then the classes are selected from the target vector.

    is_fitted : bool or str, default="auto"
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If "auto" (default), a helper method will check if the estimator
        is fitted before fitting it again.

    kwargs: dict
        Keyword arguments passed to the super class. Here, used
        to colorize the bars in the histogram.

    Attributes
    ----------
    score_ : float
        Global accuracy score

    predictions_ : ndarray
        An ndarray of predictions whose rows are the true classes and
        whose columns are the predicted classes

    Notes
    -----
    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """

    def __init__(self, model, ax=None, classes=None, is_fitted="auto", **kwargs):
        super(ClassPredictionError, self).__init__(
            model, ax=ax, classes=classes, is_fitted=is_fitted, **kwargs
        )

    def score(self, X, y, **kwargs):
        """
        Generates a 2D array where each row is the count of the
        predicted classes and each column is the true class

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        Returns
        -------
        score_ : float
            Global accuracy score
        """

        # We're relying on predict to raise NotFitted
        y_pred = self.predict(X)

        y_type, y_true, y_pred = _check_targets(y, y_pred)

        if y_type not in ("binary", "multiclass"):
            raise YellowbrickValueError("{} is not supported".format(y_type))

        indices = unique_labels(y_true, y_pred)

        if len(self.classes_) > len(indices):
            raise ModelError(
                "y and y_pred contain zero values for one of the specified classes"
            )
        elif len(self.classes_) < len(indices):
            raise NotImplementedError("filtering classes is currently not supported")

        # Create a table of predictions whose rows are the true classes
        # and whose columns are the predicted classes; each element
        # is the count of predictions for that class that match the true
        # value of that class.
        self.predictions_ = np.array(
            [
                [(y_pred[y == label_t] == label_p).sum() for label_p in indices]
                for label_t in indices
            ]
        )

        self.draw()
        self.score_ = self.estimator.score(X, y)

        return self.score_

    def draw(self):
        """
        Renders the class prediction error across the axis.

        Returns
        -------
        ax : Matplotlib Axes
            The axes on which the figure is plotted
        """

        legend_kws = {"bbox_to_anchor": (1.04, 0.5), "loc": "center left"}
        bar_stack(
            self.predictions_,  # TODO: if not fitted, should raise a NotFitted error
            self.ax,
            labels=list(self.classes_),
            ticks=self.classes_,
            colors=self.colors,
            legend_kws=legend_kws,
        )
        return self.ax

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.
        """

        # Set the title
        self.set_title("Class Prediction Error for {}".format(self.name))

        # Set the axes labels
        self.ax.set_xlabel("actual class")
        self.ax.set_ylabel("number of predicted class")

        # Compute the ceiling for the y limit
        cmax = max([sum(predictions) for predictions in self.predictions_])
        self.ax.set_ylim(0, cmax + cmax * 0.1)

        # Ensure the legend fits on the figure
        plt.tight_layout(rect=[0, 0, 0.90, 1])  # TODO: Could use self.fig now


##########################################################################
## Quick Method
##########################################################################


def class_prediction_error(
    model,
    X,
    y=None,
    ax=None,
    classes=None,
    test_size=0.2,
    random_state=None,
    is_fitted="auto",
    **kwargs
):
    """Quick method:
    Divides the dataset X and y into train and test splits, fits the model on the train
    split, then scores the model on the test split. The visualizer displays the support
    for each class in the fitted classification model displayed as a stacked bar plot.
    Each bar is segmented to show the distribution of predicted classes for each class.

    This helper function is a quick wrapper to utilize the ClassPredictionError for
    one-off analysis.

    Parameters
    ----------
    model : the Scikit-Learn estimator (should be a classifier)

    X  : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features.

    y  : ndarray or Series of length n
        An array or series of target or class values.

    ax : matplotlib axes
        The axes to plot the figure on.

    classes : list of strings
        The names of the classes in the target

    test_size : float, default=0.2
        The percentage of the data to reserve as test data.

    random_state : int or None, default=None
        The value to seed the random number generator for shuffling data.

    is_fitted : bool or str, default="auto"
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If "auto" (default), a helper method will check if the estimator
        is fitted before fitting it again.

    kwargs: dict
        Keyword arguments passed to the super class. Here, used
        to colorize the bars in the histogram.

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the class prediction error plot was drawn on.
    """
    # Instantiate the visualizer
    visualizer = ClassPredictionError(
        model=model, ax=ax, classes=classes, is_fitted=is_fitted, **kwargs
    )

    # Create the train and test splits
    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=test_size, random_state=random_state
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X_train, y_train, **kwargs)
    visualizer.score(X_test, y_test)

    # Return the visualizer
    return visualizer
