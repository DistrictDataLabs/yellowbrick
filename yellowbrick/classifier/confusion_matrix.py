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

import warnings
import numpy as np

from sklearn.metrics import confusion_matrix

from ..utils import div_safe
from ..style import find_text_color
from ..style.palettes import color_sequence
from .base import ClassificationScoreVisualizer


##########################################################################
## ConfusionMatrix
##########################################################################

CMAP_UNDERCOLOR = 'w'
CMAP_OVERCOLOR = '#2a7d4f'


class ConfusionMatrix(ClassificationScoreVisualizer):
    """
    Creates a heatmap visualization of the sklearn.metrics.confusion_matrix().
    A confusion matrix shows each combination of the true and predicted
    classes for a test data set.

    The default color map uses a yellow/orange/red color scale. The user can
    choose between displaying values as the percent of true (cell value
    divided by sum of row) or as direct counts. If percent of true mode is
    selected, 100% accurate predictions are highlighted in green.

    Requires a classification model.

    Parameters
    ----------
    model : estimator
        Must be a classifier, otherwise raises YellowbrickTypeError

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    sample_weight: array-like of shape = [n_samples], optional
        Passed to ``confusion_matrix`` to weight the samples.

    percent: bool, default: False
        Determines whether or not the confusion_matrix is displayed as counts
        or as a percent of true predictions. Note, if specifying a subset of
        classes, percent should be set to False or inaccurate figures will be
        displayed.

    classes : list, default: None
        a list of class names to use in the confusion_matrix.
        This is passed to the ``labels`` parameter of
        ``sklearn.metrics.confusion_matrix()``, and follows the behaviour
        indicated by that function. It may be used to reorder or select a
        subset of labels. If None, classes that appear at least once in
        ``y_true`` or ``y_pred`` are used in sorted order.

    label_encoder : dict or LabelEncoder, default: None
        When specifying the ``classes`` argument, the input to ``fit()``
        and ``score()`` must match the expected labels. If the ``X`` and ``y``
        datasets have been encoded prior to training and the labels must be
        preserved for the visualization, use this argument to provide a
        mapping from the encoded class to the correct label. Because typically
        a Scikit-Learn ``LabelEncoder`` is used to perform this operation, you
        may provide it directly to the class to utilize its fitted encoding.

    cmap : string, default: ``'YlOrRd'``
        Specify a colormap to define the heatmap of the predicted class
        against the actual class in the confusion matrix.

    Attributes
    ----------
    confusion_matrix_ : array, shape = [n_classes, n_classes]
        The numeric scores of the confusion matrix

    class_counts_ : array, shape = [n_classes,]
        The total number of each class supporting the confusion matrix

    Examples
    --------
    >>> from yellowbrick.classifier import ConfusionMatrix
    >>> from sklearn.linear_model import LogisticRegression
    >>> viz = ConfusionMatrix(LogisticRegression())
    >>> viz.fit(X_train, y_train)
    >>> viz.score(X_test, y_test)
    >>> viz.poof()
    """


    def __init__(self, model, ax=None, classes=None, sample_weight=None,
                 percent=False, label_encoder=None, cmap='YlOrRd', **kwargs):
        super(ConfusionMatrix, self).__init__(
            model, ax=ax, classes=classes, **kwargs
        )

        # Visual parameters
        self.cmap = color_sequence(cmap)
        self.cmap.set_under(color=CMAP_UNDERCOLOR)
        self.cmap.set_over(color=CMAP_OVERCOLOR)

        # Estimator parameters
        self.label_encoder = label_encoder
        self.sample_weight = sample_weight
        self.percent = percent

        # Used to draw diagonal line for predicted class = true class
        self._edgecolors = []

    def score(self, X, y, **kwargs):
        """
        Draws a confusion matrix based on the test data supplied by comparing
        predictions on instances X with the true values specified by the
        target vector y.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values
        """
        # Perform deprecation warnings for attributes to score
        # TODO: remove this in v0.9
        for param in ("percent", "sample_weight"):
            if param in kwargs:
                msg = (
                    "specifying '{}' in score is no longer supported, "
                    "pass to constructor of the visualizer instead."
                ).format(param)
                warnings.warn(msg, DeprecationWarning)

                setattr(self, param, kwargs[param])

        # Create predictions from X (will raise not fitted error)
        y_pred = self.predict(X)

        # Encode the target with the supplied label encoder
        if self.label_encoder:
            try :
                y = self.label_encoder.inverse_transform(y)
                y_pred = self.label_encoder.inverse_transform(y_pred)
            except AttributeError:
                # if a mapping is passed to class apply it here.
                y = [self.label_encoder[x] for x in y]
                y_pred = [self.label_encoder[x] for x in y_pred]

        # Compute the confusion matrix and class counts
        self.confusion_matrix_ = confusion_matrix(
            y, y_pred, labels=self.classes_, sample_weight=self.sample_weight
        )
        self.class_counts_ = self.class_counts(y)

        # Make array of only the classes actually being used.
        # Needed because sklearn confusion_matrix only returns counts for
        # selected classes but percent should be calculated on all classes
        selected_class_counts = []
        for c in self.classes_:
            try:
                selected_class_counts.append(self.class_counts_[c])
            except KeyError:
                selected_class_counts.append(0)
        self.class_counts_ = np.array(selected_class_counts)

        return self.draw()

    def draw(self):
        """
        Renders the classification report; must be called after score.
        """
        if self.percent == True:
            # Convert confusion matrix to percent of each row, i.e. the
            # predicted as a percent of true in each class.
            # Note: div_safe function returns 0 instead of NAN.
            self._confusion_matrix_display = div_safe(
                    self.confusion_matrix_,
                    self.class_counts_
                )
            self._confusion_matrix_display = np.round(
                self._confusion_matrix_display* 100, decimals=0
            )
        else:
            self._confusion_matrix_display = self.confusion_matrix_

        # Y axis should be sorted top to bottom in pcolormesh
        self._confusion_matrix_plottable = self._confusion_matrix_display[::-1,::]

        #Set up the dimensions of the pcolormesh
        X = np.linspace(start=0, stop=len(self.classes_), num=len(self.classes_)+1)
        Y = np.linspace(start=0, stop=len(self.classes_), num=len(self.classes_)+1)
        self.ax.set_ylim(bottom=0, top=self._confusion_matrix_plottable.shape[0])
        self.ax.set_xlim(left=0, right=self._confusion_matrix_plottable.shape[1])

        #Put in custom axis labels
        xticklabels = self.classes_
        yticklabels = self.classes_[::-1]
        xticks = np.arange(0, len(self.classes_), 1) + .5
        yticks = np.arange(0, len(self.classes_), 1) + .5
        self.ax.set(xticks=xticks, yticks=yticks)
        self.ax.set_xticklabels(xticklabels, rotation="vertical")
        self.ax.set_yticklabels(yticklabels)

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
                scaled_grid_val = grid_val / self._confusion_matrix_plottable.max()
                base_color = self.cmap(scaled_grid_val)
                text_color= find_text_color(base_color)

                #make zero values more subtle
                if self._confusion_matrix_plottable[x_int,y_int] == 0:
                    text_color = "0.75"

                #Put the data labels in the middle of the heatmap square
                self.ax.text(text_y,
                            text_x,
                            "{:.0f}{}".format(grid_val,"%" if self.percent==True else ""),
                            va='center',
                            ha='center',
                            color=text_color)

                #If the prediction is correct, put a bounding box around that square to better highlight it to the user
                #This will be used in ax.pcolormesh, setting now since we're iterating over the matrix
                    #ticklabels are conveniently already reversed properly to match the _confusion_matrix_plottalbe order
                if xticklabels[x_int] == yticklabels[y_int]:
                    self._edgecolors.append('black')
                else:
                    self._edgecolors.append('w')

        # Draw the heatmap. vmin and vmax operate in tandem with the cmap.set_under and cmap.set_over to alter the color of 0 and 100
        highest_count = self._confusion_matrix_plottable.max()
        vmax = 99.999 if self.percent == True else highest_count
        self.ax.pcolormesh(X, Y,
            self._confusion_matrix_plottable,
            vmin=0.00001,
            vmax=vmax,
            edgecolor=self._edgecolors,
            cmap=self.cmap,
            linewidth='0.01') #edgecolor='0.75', linewidth='0.01'
        return self.ax

    def finalize(self, **kwargs):
        self.set_title('{} Confusion Matrix'.format(self.name))
        self.ax.set_ylabel('True Class')
        self.ax.set_xlabel('Predicted Class')
