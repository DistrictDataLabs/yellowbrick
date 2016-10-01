# yellowbrick.classifier
# Visualizations related to evaluating Scikit-Learn classification models
#
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

from .color_utils import ddlheatmap
from .utils import get_model_name, isestimator
from .base import Visualizer, ScoreVisualizer, MultiModelMixin

##########################################################################
## Classification Visualization Base Object
##########################################################################

class ClassificationScoreVisualizer(ScoreVisualizer):

    def __init__(self, model):
        """
        Check to see if model is an instance of a classifer.
        Should return a metrics mismatch error if it isn't.
        """
        pass

##########################################################################
## Classification Report
##########################################################################

class ClassificationReport(ClassificationScoreVisualizer):
    """
    Classification report that shows the precision, recall, and F1 scores
    for the model. Integrates numerical scores as well color-coded heatmap.
    """
    def __init__(self, model, **kwargs):
        """
        Pass in a fitted model to generate a ROC curve.
        """
        self.estimator = model
        self.name = get_model_name(self.estimator)
        self.cmap = kwargs.pop('cmap', ddlheatmap)
        self.classes = model.classes_


    def score(self, y, y_pred=None, **kwargs):
        """
        Generates the Scikit-Learn classification_report
        """
        self.keys = ('precision', 'recall', 'f1')
        self.scores = precision_recall_fscore_support(y, y_pred, labels=self.classes)
        self.scores = map(lambda s: dict(zip(self.classes, s)), self.scores[0:3])
        self.scores = dict(zip(self.keys, self.scores))
        self._draw(y, y_pred)


    def _draw(self, y, y_pred):
        """
        Renders the classification report across each axis.
        """
        fig, ax = plt.subplots(1)

        self.matrix = []
        for cls in self.classes:
            self.matrix.append([self.scores['precision'][cls],self.scores['recall'][cls],self.scores['f1'][cls]])

        for column in range(len(self.matrix)+1):
            for row in range(len(self.classes)):
                ax.text(column,row,self.matrix[row][column],va='center',ha='center')

        fig = plt.imshow(self.matrix, interpolation='nearest', cmap=self.cmap)
        return ax


    def poof(self):
        """
        Plots a classification report as a heatmap.
        """
        plt.title('{} Classification Report'.format(self.name))
        plt.colorbar()
        x_tick_marks = np.arange(len(self.classes)+1)
        y_tick_marks = np.arange(len(self.classes))
        plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
        plt.yticks(y_tick_marks, self.classes)
        plt.ylabel('Classes')
        plt.xlabel('Measures')

        return plt


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
        self.estimator = model
        self.name = get_model_name(self.estimator)
        super(ROCAUC, self).__init__(model, **kwargs)
        self.colors = {
            'roc': kwargs.pop('roc_color', '#2B94E9'),
            'diagonal': kwargs.pop('diagonal_color', '#666666'),
        }


    def fit(self):
        pass

    def predict(self):
        pass

    def score(self, y, y_pred=None):
        self.fpr, self.tpr, self.thresholds = roc_curve(y, y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        self._draw(y, y_pred)

    def _draw(self, y, y_pred):
        """
        Renders ROC-AUC plot.
        Called internally by score, possibly more than once
        """
        plt.figure()
        plt.plot(self.fpr, self.tpr, c=self.colors['roc'], label='AUC = {:0.2f}'.format(self.roc_auc))

        # Plot the line of no discrimination to compare the curve to.
        plt.plot([0,1],[0,1],'m--',c=self.colors['diagonal'])


    def poof(self, **kwargs):
        """
        Called by user.

        Only takes self.

        Take in the model as input and generates a plot of
        the ROC plots with AUC metrics embedded.
        """
        plt.title('ROC for {}'.format(self.name))
        plt.legend(loc='lower right')

        plt.xlim([-0.02,1])
        plt.ylim([0,1.1])

        return plt
