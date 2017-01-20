# yellowbrick.classifier
# Visualizations related to evaluating Scikit-Learn classification models
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Wed May 18 12:39:40 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: classifier.py [5eee25b] benjamin@bengfort.com $

"""
Visualizations related to evaluating Scikit-Learn classification models
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support

from .exceptions import YellowbrickTypeError
from .utils import get_model_name, isestimator, isclassifier
from .base import Visualizer, ScoreVisualizer, MultiModelMixin
from .style.palettes import color_sequence, color_palette, LINE_COLOR


##########################################################################
## Classification Visualization Base Object
##########################################################################

class ClassificationScoreVisualizer(ScoreVisualizer):

    def __init__(self, model, ax=None, **kwargs):
        """
        Check to see if model is an instance of a classifer.
        Should return an error if it isn't.
        """
        if not isclassifier(model):
            raise YellowbrickTypeError(
                "This estimator is not a classifier; try a regression or clustering score visualizer instead!"
        )

        super(ClassificationScoreVisualizer, self).__init__(model, ax=ax, **kwargs)


##########################################################################
## Classification Report
##########################################################################

class ClassificationReport(ClassificationScoreVisualizer):
    """
    Classification report that shows the precision, recall, and F1 scores
    for the model. Integrates numerical scores as well color-coded heatmap.

    """
    def __init__(self, model, ax=None, classes=None, **kwargs):
        """
        Pass in a fitted model to generate a classification report.

        Parameters
        ----------

        :param ax: the axis to plot the figure on.

        :param model: the Scikit-Learn estimator
            Should be an instance of a classifier, else the __init__ will
            return an error.

        :param classes: a list of class names for the legend
            If classes is None and a y value is passed to fit then the classes
            are selected from the target vector.

        :param colormap: optional string or matplotlib cmap to colorize lines
            Use sequential heatmap.

        :param kwargs: keyword arguments passed to the super class.

        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.
        """
        super(ClassificationReport, self).__init__(model, ax=ax, **kwargs)

        ## hoisted to ScoreVisualizer base class
        self.estimator = model
        self.name = get_model_name(self.estimator)

        self.cmap = color_sequence(kwargs.pop('cmap', 'YlOrRd'))
        self.classes_ = classes

    def fit(self, X, y=None, **kwargs):
        """
        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: keyword arguments passed to Scikit-Learn API.
        """
        super(ClassificationReport, self).fit(X, y, **kwargs)
        if self.classes_ is None:
            self.classes_ = self.estimator.classes_
        return self

    def score(self, X, y=None, **kwargs):
        """
        Generates the Scikit-Learn classification_report

        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        """
        y_pred = self.predict(X)
        keys   = ('precision', 'recall', 'f1')
        self.scores = precision_recall_fscore_support(y, y_pred)
        self.scores = map(lambda s: dict(zip(self.classes_, s)), self.scores[0:3])
        self.scores = dict(zip(keys, self.scores))
        return self.draw(y, y_pred)

    def draw(self, y, y_pred):
        """
        Renders the classification report across each axis.

        Parameters
        ----------

        y : ndarray or Series of length n
            An array or series of target or class values

        y_pred : ndarray or Series of length n
            An array or series of predicted target values
        """
        # Create the axis if it doesn't exist
        if self.ax is None:
            self.ax = plt.gca()

        self.matrix = []
        for cls in self.classes_:
            self.matrix.append([self.scores['precision'][cls],self.scores['recall'][cls],self.scores['f1'][cls]])

        for column in range(len(self.matrix)+1):
            for row in range(len(self.classes_)):
                self.ax.text(column,row,self.matrix[row][column],va='center',ha='center')

        fig = plt.imshow(self.matrix, interpolation='nearest', cmap=self.cmap, vmin=0, vmax=1)

        return self.ax

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """
        # Set the title of the classifiation report
        self.set_title('{} Classification Report'.format(self.name))

        # Add the color bar
        plt.colorbar()

        # Compute the tick marks for both x and y
        x_tick_marks = np.arange(len(self.classes_)+1)
        y_tick_marks = np.arange(len(self.classes_))

        # Set the tick marks appropriately
        # TODO: make sure this goes through self.ax not plt
        plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
        plt.yticks(y_tick_marks, self.classes_)

        # Set the labels for the two axes
        self.ax.set_ylabel('Classes')
        self.ax.set_xlabel('Measures')


def classification_report(model, X, y=None, ax=None, classes=None, **kwargs):
    """Quick method:

    Displays precision, recall, and F1 scores for the model.
    Integrates numerical scores as well color-coded heatmap.

    This helper function is a quick wrapper to utilize the ClassificationReport
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
        Returns the axes that the classification report was drawn on.
    """
    # Instantiate the visualizer
    visualizer = ClassificationReport(model, ax, classes, **kwargs)

    # Create the train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X_train, y_train, **kwargs)
    visualizer.score(X_test, y_test)

    # Return the axes object on the visualizer
    return visualizer.ax

##########################################################################
## Receiver Operating Characteristics
##########################################################################

class ROCAUC(ClassificationScoreVisualizer):
    """
    Plot the ROC to visualize the tradeoff between the classifier's
    sensitivity and specificity.
    """
    def __init__(self, model, ax=None, **kwargs):
        """
        Pass in a fitted model to generate a ROC curve.

        Parameters
        ----------

        :param ax: the axis to plot the figure on.

        :param model: the Scikit-Learn estimator
            Should be an instance of a classifier, else the __init__ will
            return an error.

        :param roc_color: color of the ROC curve
            Specify the color as a matplotlib color: you can specify colors in
            many weird and wonderful ways, including full names ('green'), hex
            strings ('#008000'), RGB or RGBA tuples ((0,1,0,1)) or grayscale
            intensities as a string ('0.8').

        :param diagonal_color: color of the diagonal
            Specify the color as a matplotlib color.

        :param kwargs: keyword arguments passed to the super class.
            Currently passing in hard-coded colors for the Receiver Operating
            Characteristic curve and the diagonal.
            These will be refactored to a default Yellowbrick style.

        These parameters can be influenced later on in the visualization
        process, but can and should be set as early as possible.

        """
        super(ROCAUC, self).__init__(model, ax=ax, **kwargs)

        ## hoisted to ScoreVisualizer base class
        self.name = get_model_name(self.estimator)

        # Color map defaults as follows:
        # ROC color is the current color in the cycle
        # Diagonal color is the default LINE_COLOR
        self.colors = {
            'roc': kwargs.pop('roc_color', None),
            'diagonal': kwargs.pop('diagonal_color', LINE_COLOR),
        }

    def score(self, X, y=None, **kwargs):
        """
        Generates the predicted target values using the Scikit-Learn
        estimator.

        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        Returns
        ------

        ax : the axis with the plotted figure

        """
        y_pred = self.predict(X)
        self.fpr, self.tpr, self.thresholds = roc_curve(y, y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        return self.draw(y, y_pred)

    def draw(self, y, y_pred):
        """
        Renders ROC-AUC plot.
        Called internally by score, possibly more than once

        Parameters
        ----------

        y : ndarray or Series of length n
            An array or series of target or class values

        y_pred : ndarray or Series of length n
            An array or series of predicted target values

        Returns
        ------

        ax : the axis with the plotted figure

        """
        # Create the axis if it doesn't exist
        if self.ax is None:
            self.ax = plt.gca()

        plt.plot(self.fpr, self.tpr, c=self.colors['roc'], label='AUC = {:0.2f}'.format(self.roc_auc))

        # Plot the line of no discrimination to compare the curve to.
        plt.plot([0,1],[0,1],'m--',c=self.colors['diagonal'])

        return self.ax

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        """
        # Set the title and add the legend
        self.set_title('ROC for {}'.format(self.name))
        self.ax.legend(loc='lower right')

        # Set the limits for the ROC/AUC (always between 0 and 1)
        self.ax.set_xlim([-0.02, 1.0])
        self.ax.set_ylim([ 0.00, 1.1])


def roc_auc(model, X, y=None, ax=None, **kwargs):
    """Quick method:

    Displays the tradeoff between the classifier's
    sensitivity and specificity.

    This helper function is a quick wrapper to utilize the ROCAUC
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

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the roc-auc curve was drawn on.
    """
    # Instantiate the visualizer
    visualizer = ROCAUC(model, ax, **kwargs)

    # Create the train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X_train, y_train, **kwargs)
    visualizer.score(X_test, y_test)

    # Return the axes object on the visualizer
    return visualizer.ax


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

    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """
    def __init__(self, model, ax=None, classes=None, **kwargs):

        super(ClassBalance, self).__init__(model, ax=ax, **kwargs)

        self.colors    = color_palette(kwargs.pop('colors', None))
        self.classes_  = classes

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
        super(ClassBalance, self).fit(X, y, **kwargs)
        if self.classes_ is None:
            self.classes_ = self.estimator.classes_
        return self

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
        # Create the axis if it doesn't exist
        if self.ax is None:
            self.ax = plt.gca()

        #TODO: Would rather not have to set the colors with this method.
        # Refactor to make better use of yb_palettes module?

        colors = self.colors[0:len(self.classes_)]
        plt.bar(np.arange(len(self.support)), self.support.values(), color=colors, align='center', width=0.5)

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
        # TODO: change to the self.ax method rather than plt.xticks
        plt.xticks(np.arange(len(self.support)), self.support.keys())

        # Compute the ceiling for the y limit
        cmax, cmin = max(self.support.values()), min(self.support.values())
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
