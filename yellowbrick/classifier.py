# yellowbrick.classifier
# Visualizations related to evaluating Scikit-Learn classification models
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Wed May 18 12:39:40 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: classifier.py [] benjamin@bengfort.com $

"""
Visualizations related to evaluating Scikit-Learn classification models
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

from .color_utils import ddlheatmap
from .utils import get_model_name, isestimator
from .base import ModelVisualization, MultiModelMixin


##########################################################################
## Classification Visualization Base Object
##########################################################################

class ClassifierVisualization(ModelVisualization):
    pass

##########################################################################
## Classification Report
##########################################################################

class ClassifierReport(ClassifierVisualization):
    """
    """

    def __init__(self, model, **kwargs):
        self.model = model
        self.cmap = kwargs.pop('cmap', ddlheatmap)
        self.name = kwargs.pop('name', get_model_name(model))
        self.report = None


    def parse_report(self):
        """
        Custom classification_report parsing utility
        """

        if self.report is None:
            raise ModelError("Call score() before generating the model for parsing.")

        # TODO: make a bit more robust, or look for the sklearn util that doesn't stringify
        lines = self.report.split('\n')
        classes = []
        matrix = []

        for line in lines[2:(len(lines)-3)]:
            s = line.split()
            classes.append(s[0])
            value = [float(x) for x in s[1: len(s) - 1]]
            matrix.append(value)

        return matrix, classes


    def score(self, y_true, y_pred, **kwargs):
        """
        generates the Scikit-Learn classification_report
        """
        # TODO: Do a better job of guessing defaults from the model
        cr_kwargs = {
            'labels': kwargs.pop('labels', None),
            'target_names': kwargs.pop('target_names', None),
            'sample_weight': kwargs.pop('sample_weight', None),
            'digits': kwargs.pop('digits', 2)
        }

        self.report = classification_report(y_true, y_pred, **cr_kwargs)


    def render(self):
        title  = '{} Classification Report'.format(self.name)
        matrix, classes = self.parse_report()

        fig, ax = plt.subplots(1)

        for column in range(len(matrix)+1):
            for row in range(len(classes)):
                txt = matrix[row][column]
                ax.text(column,row,matrix[row][column],va='center',ha='center')

        fig = plt.imshow(matrix, interpolation='nearest', cmap=self.cmap)
        plt.title(title)
        plt.colorbar()
        x_tick_marks = np.arange(len(classes)+1)
        y_tick_marks = np.arange(len(classes))
        plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
        plt.yticks(y_tick_marks, classes)
        plt.ylabel('Classes')
        plt.xlabel('Measures')

        return ax


def crplot(model, y_true, y_pred, **kwargs):
    """
    Plots a classification report as a heatmap. (More to follow).
    """
    viz = ClassifierReport(model, **kwargs)
    viz.score(y_true, y_pred, **kwargs)

    return viz.render()


##########################################################################
## Receiver Operating Characteristics
##########################################################################

class ROCAUC(MultiModelMixin, ClassifierVisualization):
    """
    """
    def __init__(self, models, **kwargs):
        """
        """
        super(ROCAUC, self).__init__(models, **kwargs)
        self.colors = {
            'roc': kwargs.pop('roc_color', '#2B94E9'),
            'diagonal': kwargs.pop('diagonal_color', '#666666'),
        }

    def fit(self, X, y):
        self.models = list(map(lambda model: model.fit(X, y), self.models))

    def render(self, X, y):

        for idx, axe in enumerate(self.generate_subplots()):
            # Get the information for this axis
            name  = self.names[idx]
            model = self.models[idx]
            y_pred = model.predict(X)
            fpr, tpr, thresholds = roc_curve(y, y_pred)
            roc_auc = auc(fpr, tpr)

            axe.plot(fpr, tpr, c=self.colors['roc'], label='AUC = {:0.2f}'.format(roc_auc))

            # Plot the line of no discrimination to compare the curve to.
            axe.plot([0,1],[0,1],'m--',c=self.colors['diagonal'])

            axe.set_title('ROC for {}'.format(name))
            axe.legend(loc='lower right')

        plt.xlim([0,1])
        plt.ylim([0,1.1])

        return axe


def rocplot(models, X, y, **kwargs):
    """
    Plots a ROC plot with AUC metric embedded.
    """
    viz = ROCAUC(models, **kwargs)
    viz.fit(X, y)

    return viz.render(X, y)
