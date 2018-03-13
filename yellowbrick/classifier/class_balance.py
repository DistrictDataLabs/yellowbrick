# yellowbrick.classifier.class_balance
# Class balance visualizer for showing per-class support.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Author:   Neal Humphrey
# Author:   Larry Gray
# Created:  Wed May 18 12:39:40 2016 -0400
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: class_balance.py [5388065] neal@nhumphrey.com $

"""
Class balance visualizer for showing per-class support.
"""

##########################################################################
## Imports
##########################################################################

import matplotlib.pyplot as plt
import numpy as np

from .base import ClassificationScoreVisualizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.classification import _check_targets

from ..exceptions import ModelError, YellowbrickValueError
from ..style.colors import resolve_colors


##########################################################################
## Class Balance Chart
##########################################################################

class ClassBalance(ClassificationScoreVisualizer):
    """
    Class balance chart that shows the support for each class in the
    fitted classification model displayed as a bar plot. It is initialized
    with a fitted model and generates a class balance chart on draw.

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

    Notes
    -----
    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """

    def score(self, X, y=None, **kwargs):
        """
        Generates the Scikit-Learn precision_recall_fscore_support

        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        Returns
        -------

        ax : the axis with the plotted figure
        """
        y_pred = self.predict(X)
        self.scores  = precision_recall_fscore_support(y, y_pred)
        self.support = dict(zip(self.classes_, self.scores[-1]))
        return self.draw()

    def draw(self):
        """
        Renders the class balance chart across the axis.

        Returns
        -------
        ax : the axis with the plotted figure

        """
        #TODO: Would rather not have to set the colors with this method.
        # Refactor to make better use of yb_palettes module?

        colors = self.colors[0:len(self.classes_)]
        self.ax.bar(
            np.arange(len(self.support)), self.support.values(),
            color=colors, align='center', width=0.5
        )

        return self.ax

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """
        # Set the title
        self.set_title('Class Balance for {}'.format(self.name))

        # Set the x ticks with the class names
        self.ax.set_xticks(np.arange(len(self.support)))
        self.ax.set_xticklabels(self.support.keys())

        # Compute the ceiling for the y limit
        cmax = max(self.support.values())
        self.ax.set_ylim(0, cmax + cmax* 0.1)


def class_balance(model, X, y=None, ax=None, classes=None, **kwargs):
    """Quick method:

    Displays the support for each class in the
    fitted classification model displayed as a bar plot.

    This helper function is a quick wrapper to utilize the ClassBalance
    ScoreVisualizer for one-off analysis.

    Parameters
    ----------
    X  : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features.

    y  : ndarray or Series of length n
        An array or series of target or class values.

    ax : matplotlib axes
        The axes to plot the figure on.

    model : the Scikit-Learn estimator (should be a classifier)

    classes : list of strings
        The names of the classes in the target

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the class balance plot was drawn on.
    """
    # Instantiate the visualizer
    visualizer = ClassBalance(model, ax, classes, **kwargs)

    # Create the train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X_train, y_train, **kwargs)
    visualizer.score(X_test, y_test)

    # Return the axes object on the visualizer
    return visualizer.ax


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

        ax : the axis with the plotted figure
        """

        # We're replying on predict to raise NotFitted
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

        # Create a table of scores whose rows are the true classes
        # and whose columns are the predicted classes; each element
        # is the count of predictions for that class that match the true
        # value of that class.
        self.scores_ = np.array([
            [
                (y_pred[y == label_t] == label_p).sum()
                for label_p in indices
            ]
            for label_t in indices
        ])

        return self.draw()

    def draw(self):
        """
        Renders the class prediction error across the axis.
        Returns
        -------
        ax : the axis with the plotted figure
        """

        indices = np.arange(len(self.classes_))
        prev = np.zeros(len(self.classes_))

        colors = resolve_colors(
            colors=self.colors,
            n_colors=len(self.classes_))

        for idx, row in enumerate(self.scores_):
            self.ax.bar(indices, row, label=self.classes_[idx],
                        bottom=prev, color=colors[idx])
            prev += row

        return self.ax

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.
        Parameters
        ----------
        kwargs: generic keyword arguments.

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
        cmax = max([sum(scores) for scores in self.scores_])
        self.ax.set_ylim(0, cmax + cmax * 0.1)

        # Put the legend outside of the graph
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.tight_layout(rect=[0, 0, 0.85, 1])


def class_prediction_error(model, X, y=None, ax=None, classes=None,
                           test_size=0.2, **kwargs):
    """Quick method:
    Displays the support for each class in the
    fitted classification model displayed as a stacked bar plot.
    Each bar is segmented to show the distribution of predicted
    classes for each class.

    This helper function is a quick wrapper to utilize the ClassPredictionError
    ScoreVisualizer for one-off analysis.
    Parameters
    ----------
    X  : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features.
    y  : ndarray or Series of length n
        An array or series of target or class values.
    ax : matplotlib axes
        The axes to plot the figure on.
    model : the Scikit-Learn estimator (should be a classifier)
    classes : list of strings
        The names of the classes in the target
    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the class prediction error plot was drawn on.
    """
    # Instantiate the visualizer
    visualizer = ClassPredictionError(model, ax, classes, **kwargs)

    # Create the train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size)

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X_train, y_train, **kwargs)
    visualizer.score(X_test, y_test)

    # Return the axes object on the visualizer
    return visualizer.ax
