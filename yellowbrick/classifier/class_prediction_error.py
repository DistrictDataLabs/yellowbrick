# yellowbrick.classifier.class_prediction_error
# Shows the balance of classes and their associated predictions.
#
# Author:   Larry Gray
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Wed May 18 12:39:40 2016 -0400
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

from .base import ClassificationScoreVisualizer

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.classification import _check_targets
from sklearn.model_selection import train_test_split as tts

from ..exceptions import ModelError, YellowbrickValueError
from ..style.colors import resolve_colors


##########################################################################
## Class Prediction Error Chart
##########################################################################

class ClassPredictionError(ClassificationScoreVisualizer):
    """
    Class Prediction Error chart that shows the support for each class in the
    fitted classification model displayed as a stacked bar.  Each bar is
    segmented to show the distribution of predicted classes for each
    class. It is initialized with a fitted model and generates a
    class prediction error chart on draw.

    Parameters
    ----------
    ax: axes
        the axis to plot the figure on.

    model: estimator
        Scikit-Learn estimator object. Should be an instance of a classifier,
        else ``__init__()`` will raise an exception.

    classes: list
        A list of class names for the legend. If classes is None and a y value
        is passed to fit then the classes are selected from the target vector.

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
            raise YellowbrickValueError("%s is not supported" % y_type)

        indices = unique_labels(y_true, y_pred)

        if len(self.classes_) > len(indices):
            raise ModelError("y and y_pred contain zero values "
                             "for one of the specified classes")
        elif len(self.classes_) < len(indices):
            raise NotImplementedError("filtering classes is "
                                        "currently not supported")

        # Create a table of predictions whose rows are the true classes
        # and whose columns are the predicted classes; each element
        # is the count of predictions for that class that match the true
        # value of that class.
        self.predictions_ = np.array([
            [
                (y_pred[y == label_t] == label_p).sum()
                for label_p in indices
            ]
            for label_t in indices
        ])

        self.draw()
        self.score_ = self.estimator.score(X, y)

        return self.score_

    def draw(self):
        """
        Renders the class prediction error across the axis.
        """

        indices = np.arange(len(self.classes_))
        prev = np.zeros(len(self.classes_))

        colors = resolve_colors(
            colors=self.colors,
            n_colors=len(self.classes_))

        for idx, row in enumerate(self.predictions_):
            self.ax.bar(indices, row, label=self.classes_[idx],
                        bottom=prev, color=colors[idx])
            prev += row

        return self.ax

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.
        """

        indices = np.arange(len(self.classes_))

        # Set the title
        self.set_title("Class Prediction Error for {}".format(self.name))

        # Set the x ticks with the class names
        self.ax.set_xticks(indices)
        self.ax.set_xticklabels(self.classes_)

        # Set the axes labels
        self.ax.set_xlabel("actual class")
        self.ax.set_ylabel("number of predicted class")

        # Compute the ceiling for the y limit
        cmax = max([sum(predictions) for predictions in self.predictions_])
        self.ax.set_ylim(0, cmax + cmax * 0.1)

        # Put the legend outside of the graph
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.tight_layout(rect=[0, 0, 0.85, 1])


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
    **kwargs):
    """Quick method:
    Divides the dataset X and y into train and test splits, fits the model on
    the train split, then scores the model on the test split. The visualizer
    displays the support for each class in the fitted classification model
    displayed as a stacked bar plot Each bar is segmented to show the
    distribution of predicted classes for each class.

    This helper function is a quick wrapper to utilize the ClassPredictionError
    ScoreVisualizer for one-off analysis.

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

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the class prediction error plot was drawn on.
    """
    # Instantiate the visualizer
    visualizer = ClassPredictionError(model, ax, classes, **kwargs)

    # Create the train and test splits
    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=test_size, random_state=random_state
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X_train, y_train, **kwargs)
    visualizer.score(X_test, y_test)

    # Return the axes object on the visualizer
    return visualizer.ax
