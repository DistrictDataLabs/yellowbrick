# yellowbrick.classifier.confusion_matrix
# Visual confusion matrix for classifier scoring.
#
# Author:   Neal Humphrey
# Created:  Tue May 03 11:05:11 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: confusion_matrix.py [5388065] neal@nhumphrey.com $

"""
Visual confusion matrix for classifier scoring.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from sklearn.metrics import confusion_matrix

from ..utils import div_safe
from ..style import find_text_color
from ..style.palettes import color_sequence
from .base import ClassificationScoreVisualizer


##########################################################################
## ConfusionMatrix
##########################################################################

CMAP_OVERCOLOR = '#2a7d4f'


class ConfusionMatrix(ClassificationScoreVisualizer):
    """
    Creates a heatmap visualization of the sklearn.metrics.confusion_matrix(). A confusion
    matrix shows each combination of the true and predicted classes for a test data set.

    The default color map uses a yellow/orange/red color scale. The user can choose between
    displaying values as the percent of true (cell value divided by sum of row) or as direct
    counts. If percent of true mode is selected, 100% accurate predictions are highlighted in green.

    Requires a classification model

    Parameters
    ----------
    model : the Scikit-Learn estimator
        Should be an instance of a classifier or __init__ will return an error.

    ax : the matplotlib axis to plot the figure on (if None, a new axis will be created)

    classes : list, default: None
        a list of class names to use in the confusion_matrix. This is passed to the 'labels'
        parameter of sklearn.metrics.confusion_matrix(), and follows the behaviour
        indicated by that function. It may be used to reorder or select a subset of labels.
        If None, values that appear at least once in y_true or y_pred are used in sorted order.

    label_encoder : dict or LabelEncoder, default: None
        When specifying the ``classes`` argument, the input to ``fit()`` and ``score()`` must match the
        expected labels. If the ``X`` and ``y`` datasets have been encoded prior to training and the
        labels must be preserved for the visualization, use this argument to provide a mapping from the
        encoded class to the correct label. Because typically a Scikit-Learn ``LabelEncoder`` is used to
        perform this operation, you may provide it directly to the class to utilize its fitted encoding.

    Examples
    --------

    >>> from yellowbrick.classifier import ConfusionMatrix
    >>> from sklearn.linear_model import LogisticRegression
    >>> viz = ConfusionMatrix(LogisticRegression())
    >>> viz.fit(X_train, y_train)
    >>> viz.score(X_test, y_test)
    >>> viz.poof()
    """


    def __init__(self, model, ax=None, classes=None, label_encoder=None, **kwargs):
        super(ConfusionMatrix, self).__init__(
            model, ax=ax, classes=classes, **kwargs
        )

        #Initialize all the other attributes we'll use (for coder clarity)
        self.confusion_matrix = None

        self.cmap = color_sequence(kwargs.pop('cmap', 'YlOrRd'))
        self.cmap.set_under(color = 'w')
        self.cmap.set_over(color=CMAP_OVERCOLOR)
        self.edgecolors = [] #used to draw diagonal line for predicted class = true class
        self.label_encoder = label_encoder

    def score(self, X, y, sample_weight=None, percent=True):
        """
        Generates the Scikit-Learn confusion_matrix and applies this to the appropriate axis

        Parameters
        ----------

        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        sample_weight: optional, passed to the confusion_matrix

        percent: optional, Boolean. Determines whether or not the confusion_matrix
                should be displayed as raw numbers or as a percent of the true
                predictions. Note, if using a subset of classes in __init__, percent should
                be set to False or inaccurate percents will be displayed.
        """
        y_pred = self.predict(X)


        if self.label_encoder:
            try :
                y = self.label_encoder.inverse_transform(y)
                y_pred = self.label_encoder.inverse_transform(y_pred)
            except AttributeError:
                # if a mapping is passed to class apply it here.
                y = [self.label_encoder[x] for x in y]
                y_pred = [self.label_encoder[x] for x in y_pred]

        self.confusion_matrix = confusion_matrix(
            y, y_pred, labels=self.classes_, sample_weight=sample_weight
        )
        self._class_counts = self.class_counts(y)

        #Make array of only the classes actually being used.
        #Needed because sklearn confusion_matrix only returns counts for selected classes
            #but percent should be calculated based on all classes
        selected_class_counts = []
        for c in self.classes_:
            try:
                selected_class_counts.append(self._class_counts[c])
            except KeyError:
                selected_class_counts.append(0)
        self.selected_class_counts = np.array(selected_class_counts)

        return self.draw(percent)

    def draw(self, percent=True):
        """
        Renders the classification report
        Should only be called internally, as it uses values calculated in Score
        and score calls this method.

        Parameters
        ----------

        percent:    Boolean
                    Whether the heatmap should represent "% of True" or raw counts

        """
        if percent == True:
            #Convert confusion matrix to percent of each row, i.e. the predicted as a percent of true in each class
            #div_safe function returns 0 instead of NAN.
            self._confusion_matrix_display = div_safe(
                    self.confusion_matrix,
                    self.selected_class_counts
                    )
            self._confusion_matrix_display =np.round(self._confusion_matrix_display* 100, decimals=0)
        else:
            self._confusion_matrix_display = self.confusion_matrix

        #Y axis should be sorted top to bottom in pcolormesh
        self._confusion_matrix_plottable = self._confusion_matrix_display[::-1,::]

        self.max = self._confusion_matrix_plottable.max()

        #Set up the dimensions of the pcolormesh
        X = np.linspace(start=0, stop=len(self.classes_), num=len(self.classes_)+1)
        Y = np.linspace(start=0, stop=len(self.classes_), num=len(self.classes_)+1)
        self.ax.set_ylim(bottom=0, top=self._confusion_matrix_plottable.shape[0])
        self.ax.set_xlim(left=0, right=self._confusion_matrix_plottable.shape[1])

        #Put in custom axis labels
        self.xticklabels = self.classes_
        self.yticklabels = self.classes_[::-1]
        self.xticks = np.arange(0, len(self.classes_), 1) + .5
        self.yticks = np.arange(0, len(self.classes_), 1) + .5
        self.ax.set(xticks=self.xticks, yticks=self.yticks)
        self.ax.set_xticklabels(self.xticklabels, rotation="vertical", fontsize=8)
        self.ax.set_yticklabels(self.yticklabels, fontsize=8)

        ######################
        # Add the data labels to each square
        ######################
        for x_index, x in np.ndenumerate(X):
            #np.ndenumerate returns a tuple for the index, must access first element using [0]
            x_index = x_index[0]
            for y_index, y in np.ndenumerate(Y):
                #Clean up our iterators
                #numpy doesn't like non integers as indexes; also np.ndenumerate returns tuple
                x_int = int(x)
                y_int = int(y)
                y_index = y_index[0]

                #X and Y are one element longer than the confusion_matrix. Don't want to add text for the last X or Y
                if x_index == X[-1] or y_index == Y[-1]:
                    break

                #center the text in the middle of the block
                text_x = x + 0.5
                text_y = y + 0.5

                #extract the value
                grid_val = self._confusion_matrix_plottable[x_int,y_int]

                #Determine text color
                scaled_grid_val = grid_val / self.max
                base_color = self.cmap(scaled_grid_val)
                text_color= find_text_color(base_color)

                #make zero values more subtle
                if self._confusion_matrix_plottable[x_int,y_int] == 0:
                    text_color = "0.75"

                #Put the data labels in the middle of the heatmap square
                self.ax.text(text_y,
                            text_x,
                            "{:.0f}{}".format(grid_val,"%" if percent==True else ""),
                            va='center',
                            ha='center',
                            fontsize=8,
                            color=text_color)

                #If the prediction is correct, put a bounding box around that square to better highlight it to the user
                #This will be used in ax.pcolormesh, setting now since we're iterating over the matrix
                    #ticklabels are conveniently already reversed properly to match the _confusion_matrix_plottalbe order
                if self.xticklabels[x_int] == self.yticklabels[y_int]:
                    self.edgecolors.append('black')
                else:
                    self.edgecolors.append('w')

        # Draw the heatmap. vmin and vmax operate in tandem with the cmap.set_under and cmap.set_over to alter the color of 0 and 100
        highest_count = self._confusion_matrix_plottable.max()
        vmax = 99.999 if percent == True else highest_count
        self.ax.pcolormesh(X, Y,
            self._confusion_matrix_plottable,
            vmin=0.00001,
            vmax=vmax,
            edgecolor=self.edgecolors,
            cmap=self.cmap,
            linewidth='0.01') #edgecolor='0.75', linewidth='0.01'
        return self.ax

    def finalize(self, **kwargs):
        self.set_title('{} Confusion Matrix'.format(self.name))
        self.ax.set_ylabel('True Class')
        self.ax.set_xlabel('Predicted Class')
