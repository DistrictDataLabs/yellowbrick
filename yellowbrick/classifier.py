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

from .color import ddlheatmap


##########################################################################
## Classification Report
##########################################################################

def crplot(model, y_true, y_pred, **kwargs):
    """
    Plots a classification report as a heatmap. (More to follow).
    """

    # Get classification report arguments
    # TODO: Do a better job of guessing defaults from the model
    cr_kwargs = {
        'labels': kwargs.pop('labels', None),
        'target_names': kwargs.pop('target_names', None),
        'sample_weight': kwargs.pop('sample_weight', None),
        'digits': kwargs.pop('digits', 2)
    }

    # Generate the classification report
    report = classification_report(y_true, y_pred, **cr_kwargs)
    cmap   = kwargs.pop('cmap', ddlheatmap)
    title  = kwargs.pop('title', '{} Classification Report'.format(model.__class__.__name__))


    # Parse classification report: move to it's own function
    # TODO: make a bit more robust, or look for the sklearn util that doesn't stringify
    lines = report.split('\n')
    classes = []
    matrix = []

    for line in lines[2:(len(lines)-3)]:
        s = line.split()
        classes.append(s[0])
        value = [float(x) for x in s[1: len(s) - 1]]
        matrix.append(value)

    # Generate plots and figure
    fig, ax = plt.subplots(1)

    for column in range(len(matrix)+1):
        for row in range(len(classes)):
            txt = matrix[row][column]
            ax.text(column,row,matrix[row][column],va='center',ha='center')

    fig = plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(len(classes)+1)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.ylabel('Classes')
    plt.xlabel('Measures')

    return ax


##########################################################################
## Receiver Operating Characteristics
##########################################################################

def rocplot_compare(models, y_true, y_pred, **kwargs):
    """
    Plots a side by size comparison of the ROC plot with AUC metric embedded.
    """
    if len(models) != len(y_true) and len(models) != len(y_pred):
        raise ValueError(
            "Pass in two models, two sets of target and predictions"
        )

    # Set up split subplots for the curve comparison.
    # TODO: ensure that the number of models is only 2
    fig, axes = plt.subplots(1, 2, sharey=True)

    # Zip together each plot to generate them independently.
    for model, y, yhat, ax in zip(models, y_true, y_pred, axes):

        # Figure out the name of the model
        if isinstance(model, Pipeline):
            name = model.steps[-1][1].__class__.__name__
        else:
            name = model.__class__.__name__

        fpr, tpr, thresholds = roc_curve(y, yhat)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC Curve with the specified AUC label.
        ax.plot(fpr, tpr, c='#2B94E9', label='AUC = {:0.2f}'.format(roc_auc))

        # Plot the line of no discrimination to compare the curve to.
        ax.plot([0,1],[0,1],'m--',c='#666666')

        # Set the title and create the legend.
        ax.set_title('ROC for {}'.format(name))
        ax.legend(loc='lower right')

    # Refactor the limits of the plot
    plt.xlim([0,1])
    plt.ylim([0,1.1])

    return axes
