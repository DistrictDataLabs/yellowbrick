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
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support

from .style.palettes import ddlheatmap
from .exceptions import YellowbrickTypeError
from .style.palettes import PALETTES as YELLOWBRICK_PALETTES
from .utils import get_model_name, isestimator, isclassifier
from .base import Visualizer, ScoreVisualizer, MultiModelMixin

##########################################################################
## Classification Visualization Base Object
##########################################################################

class ClassificationScoreVisualizer(ScoreVisualizer):

    def __init__(self, model, **kwargs):
        """
        Check to see if model is an instance of a classifer.
        Should return an error if it isn't.
        """
        if not isclassifier(model):
            raise YellowbrickTypeError(
                "This estimator is not a classifier; try a regression or clustering score visualizer instead!"
        )

        super(ClassificationScoreVisualizer, self).__init__(model, **kwargs)

##########################################################################
## Classification Report
##########################################################################

class ClassificationReport(ClassificationScoreVisualizer):
    """
    Classification report that shows the precision, recall, and F1 scores
    for the model. Integrates numerical scores as well color-coded heatmap.
    """
    def __init__(self, model, classes=None, **kwargs):
        """
        Pass in a fitted model to generate a ROC curve.
        """
        super(ClassificationReport, self).__init__(model, **kwargs)

        # TODO: hoist
        self.estimator = model
        self.ax = None

        self.name = get_model_name(self.estimator)
        self.cmap = kwargs.pop('cmap', ddlheatmap)
        self.classes_ = classes

    def fit(self, X, y=None, **kwargs):
        super(ClassificationReport, self).fit(X, y, **kwargs)
        if self.classes_ is None:
            self.classes_ = self.estimator.classes_
        return self

    def score(self, X, y=None, **kwargs):
        """
        Generates the Scikit-Learn classification_report
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
        """
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


    def poof(self):
        """
        Plots a classification report as a heatmap.
        """
        if self.ax is None: return

        plt.title('{} Classification Report'.format(self.name))
        plt.colorbar()
        x_tick_marks = np.arange(len(self.classes_)+1)
        y_tick_marks = np.arange(len(self.classes_))
        plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
        plt.yticks(y_tick_marks, self.classes_)
        plt.ylabel('Classes')
        plt.xlabel('Measures')

        return self.ax


##########################################################################
## Receiver Operating Characteristics
##########################################################################

class ROCAUC(ClassificationScoreVisualizer):
    """
    Plot the ROC to visualize the tradeoff between the classifier's
    sensitivity and specificity.
    """
    def __init__(self, model, **kwargs):
        """
        Pass in a model to generate a ROC curve.
        """
        super(ROCAUC, self).__init__(model, **kwargs)

        # TODO hoist to main
        self.name = get_model_name(self.estimator)
        self.ax = None

        super(ROCAUC, self).__init__(model, **kwargs)
        self.colors = {
            'roc': kwargs.pop('roc_color', '#2B94E9'),
            'diagonal': kwargs.pop('diagonal_color', '#666666'),
        }

    def score(self, X, y=None, **kwargs):
        y_pred = self.predict(X)
        self.fpr, self.tpr, self.thresholds = roc_curve(y, y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        return self.draw(y, y_pred)

    def draw(self, y, y_pred):
        """
        Renders ROC-AUC plot.
        Called internally by score, possibly more than once
        """
        if self.ax is None:
            self.ax = plt.gca()
        plt.plot(self.fpr, self.tpr, c=self.colors['roc'], label='AUC = {:0.2f}'.format(self.roc_auc))

        # Plot the line of no discrimination to compare the curve to.
        plt.plot([0,1],[0,1],'m--',c=self.colors['diagonal'])

        return self.ax

    def poof(self, **kwargs):
        """
        Called by user.

        Only takes self.

        Take in the model as input and generates a plot of
        the ROC plots with AUC metrics embedded.
        """
        if self.ax is None: return

        plt.title('ROC for {}'.format(self.name))
        plt.legend(loc='lower right')

        plt.xlim([-0.02,1])
        plt.ylim([0,1.1])

        return self.ax

##########################################################################
## Class Balance Chart
##########################################################################

class ClassBalance(ClassificationScoreVisualizer):
    """
    Class balance chart that shows the support for each class in the
    fitted classification model.
    """
    def __init__(self, model, classes=None, **kwargs):
        """
        Pass in a fitted model to generate a class balance chart.
        """

        # TODO: hoist
        self.estimator = model
        self.ax = None


        self.name      = get_model_name(self.estimator)
        self.colors    = kwargs.pop('colors', YELLOWBRICK_PALETTES['paired'])
        self.classes_  = classes

    def fit(self, X, y=None, **kwargs):
        super(ClassBalance, self).fit(X, y, **kwargs)
        if self.classes_ is None:
            self.classes_ = self.estimator.classes_
        return self

    def score(self, X, y=None, **kwargs):
        """
        Generates the Scikit-Learn precision_recall_fscore_support
        """
        y_pred = self.predict(X)
        self.scores  = precision_recall_fscore_support(y, y_pred)
        self.support = dict(zip(self.classes_, self.scores[-1]))
        return self.draw()

    def draw(self):
        """
        Renders the class balance chart across the axis.

        TODO: Would rather not have to set the colors with this method.
        Refactor to make better use of yb_palettes module?

        """
        if self.ax is None:
            self.ax = plt.gca()

        colors = self.colors[0:len(self.classes_)]
        plt.bar(np.arange(len(self.support)), self.support.values(), color=colors, align='center', width=0.5)

        return self.ax

    def poof(self):
        """
        Plots a class balance chart
        """
        if self.ax is None: return

        plt.xticks(np.arange(len(self.support)), self.support.keys())
        cmax, cmin = max(self.support.values()), min(self.support.values())
        ceiling = cmax + cmax*0.1
        span = cmax - cmin
        plt.ylim(0, ceiling)
        plt.show()

        return self.ax
