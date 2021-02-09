# yellowbrick.classifier.confusion_matrix
# Visual confusion matrix for classifier scoring.
#
# Author:   Neal Humphrey
# Created:  Tue May 03 11:05:11 2017 -0700
#
# Copyright (C) 2017 The scikit-yb developers
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

from sklearn.metrics import confusion_matrix as confusion_matrix_metric

from yellowbrick.utils import div_safe
from yellowbrick.style import find_text_color
from yellowbrick.style.palettes import color_sequence
from yellowbrick.classifier.base import ClassificationScoreVisualizer


##########################################################################
## ConfusionMatrix
##########################################################################

CMAP_UNDERCOLOR = "w"
CMAP_MUTEDCOLOR = "0.75"
CMAP_OVERCOLOR = "#2a7d4f"


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
    estimator : estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If not specified the current axes will be
        used (or generated if required).

    sample_weight: array-like of shape = [n_samples], optional
        Passed to ``confusion_matrix`` to weight the samples.

    percent: bool, default: False
        Determines whether or not the confusion_matrix is displayed as counts
        or as a percent of true predictions. Note, if specifying a subset of
        classes, percent should be set to False or inaccurate figures will be
        displayed.

    classes : list of str, defult: None
        The class labels to use for the legend ordered by the index of the sorted
        classes discovered in the ``fit()`` method. Specifying classes in this
        manner is used to change the class names to a more specific format or
        to label encoded integer classes. Some visualizers may also use this
        field to filter the visualization for specific classes. For more advanced
        usage specify an encoder rather than class labels.

    encoder : dict or LabelEncoder, default: None
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.

    cmap : string, default: ``'YlOrRd'``
        Specify a colormap to define the heatmap of the predicted class
        against the actual class in the confusion matrix.

    fontsize : int, default: None
        Specify the fontsize of the text in the grid and labels to make the
        matrix a bit easier to read. Uses rcParams font size by default.

    is_fitted : bool or str, default="auto"
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If "auto" (default), a helper method will check if the estimator
        is fitted before fitting it again.

    force_model : bool, default: False
        Do not check to ensure that the underlying estimator is a classifier. This
        will prevent an exception when the visualizer is initialized but may result
        in unexpected or unintended behavior.

    kwargs : dict
        Keyword arguments passed to the visualizer base classes.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels observed while fitting.

    class_counts_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class supporting the confusion matrix.

    score_ : float
        An evaluation metric of the classifier on test data produced when
        ``score()`` is called. This metric is between 0 and 1 -- higher scores are
        generally better. For classifiers, this score is usually accuracy, but
        ensure you check the underlying model for more details about the metric.

    confusion_matrix_ : array, shape = [n_classes, n_classes]
        The numeric scores of the confusion matrix.

    Examples
    --------
    >>> from yellowbrick.classifier import ConfusionMatrix
    >>> from sklearn.linear_model import LogisticRegression
    >>> viz = ConfusionMatrix(LogisticRegression())
    >>> viz.fit(X_train, y_train)
    >>> viz.score(X_test, y_test)
    >>> viz.show()
    """

    def __init__(
        self,
        estimator,
        ax=None,
        sample_weight=None,
        percent=False,
        classes=None,
        encoder=None,
        cmap="YlOrRd",
        fontsize=None,
        is_fitted="auto",
        force_model=False,
        **kwargs
    ):
        super(ConfusionMatrix, self).__init__(
            estimator,
            ax=ax,
            classes=classes,
            encoder=encoder,
            is_fitted=is_fitted,
            force_model=force_model,
            **kwargs
        )

        # Visual parameters
        self.fontsize = fontsize
        self.cmap = color_sequence(cmap)
        self.cmap.set_under(color=CMAP_UNDERCOLOR)
        self.cmap.set_over(color=CMAP_OVERCOLOR)

        # Estimator parameters
        self.percent = percent
        self.sample_weight = sample_weight

        # Used to draw diagonal line for predicted class = true class
        self._edgecolors = []

    def score(self, X, y):
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

        Returns
        -------

        score_ : float
            Global accuracy score
        """
        # Call super to check if fitted and to compute self.score_
        super(ConfusionMatrix, self).score(X, y)

        # Create predictions from X (will raise not fitted error)
        y_pred = self.predict(X)

        # Decode the target with the label encoder and get human readable labels
        y = self._decode_labels(y)
        y_pred = self._decode_labels(y_pred)
        labels = self._labels()
        if labels is None:
            labels = self.classes_

        # Compute the confusion matrix and class counts
        self.confusion_matrix_ = confusion_matrix_metric(
            y, y_pred, labels=labels, sample_weight=self.sample_weight
        )
        self.class_counts_ = dict(zip(*np.unique(y, return_counts=True)))

        # Make array of only the classes actually being used.
        # Needed because sklearn confusion_matrix only returns counts for
        # selected classes but percent should be calculated on all classes
        selected_class_counts = []
        for c in labels:
            try:
                selected_class_counts.append(self.class_counts_[c])
            except KeyError:
                selected_class_counts.append(0)
        self.class_counts_ = np.array(selected_class_counts)

        self.draw()
        return self.score_

    def draw(self):
        """
        Renders the classification report; must be called after score.
        """

        # Perform display related manipulations on the confusion matrix data
        cm_display = self.confusion_matrix_

        # Convert confusion matrix to percent of each row, i.e. the
        # predicted as a percent of true in each class.
        if self.percent is True:
            # Note: div_safe function returns 0 instead of NAN.
            cm_display = div_safe(
                self.confusion_matrix_, self.class_counts_.reshape(-1, 1)
            )
            cm_display = np.round(cm_display * 100, decimals=0)

        # Y axis should be sorted top to bottom in pcolormesh
        cm_display = cm_display[::-1, ::]

        # Get the human readable labels
        labels = self._labels()
        if labels is None:
            labels = self.classes_

        # Set up the dimensions of the pcolormesh
        n_classes = len(labels)
        X, Y = np.arange(n_classes + 1), np.arange(n_classes + 1)
        self.ax.set_ylim(bottom=0, top=cm_display.shape[0])
        self.ax.set_xlim(left=0, right=cm_display.shape[1])

        # Fetch the grid labels from the classes in correct order; set ticks.
        xticklabels = labels
        yticklabels = labels[::-1]
        ticks = np.arange(n_classes) + 0.5

        self.ax.set(xticks=ticks, yticks=ticks)
        self.ax.set_xticklabels(
            xticklabels, rotation="vertical", fontsize=self.fontsize
        )
        self.ax.set_yticklabels(yticklabels, fontsize=self.fontsize)

        # Set data labels in the grid enumerating over all x,y class pairs.
        # NOTE: X and Y are one element longer than the confusion matrix, so
        # skip the last element in the enumeration to label grids.
        for x in X[:-1]:
            for y in Y[:-1]:

                # Extract the value and the text label
                value = cm_display[x, y]
                svalue = "{:0.0f}".format(value)
                if self.percent:
                    svalue += "%"

                # Determine the grid and text colors
                base_color = self.cmap(value / cm_display.max())
                text_color = find_text_color(base_color)

                # Make zero values more subtle
                if cm_display[x, y] == 0:
                    text_color = CMAP_MUTEDCOLOR

                # Add the label to the middle of the grid
                cx, cy = x + 0.5, y + 0.5
                self.ax.text(
                    cy,
                    cx,
                    svalue,
                    va="center",
                    ha="center",
                    color=text_color,
                    fontsize=self.fontsize,
                )

                # Add a dark line on the grid with the diagonal. Note that the
                # tick labels have already been reversed.
                lc = "k" if xticklabels[x] == yticklabels[y] else "w"
                self._edgecolors.append(lc)

        # Draw the heatmap with colors bounded by vmin,vmax
        vmin = 0.00001
        vmax = 99.999 if self.percent is True else cm_display.max()
        self.ax.pcolormesh(
            X,
            Y,
            cm_display,
            vmin=vmin,
            vmax=vmax,
            edgecolor=self._edgecolors,
            cmap=self.cmap,
            linewidth="0.01",
        )

        # Return the axes being drawn on
        return self.ax

    def show(self, outpath=None, **kwargs):
        if outpath is not None:
            kwargs["bbox_inches"] = kwargs.get("bbox_inches", "tight")
        return super(ConfusionMatrix, self).show(outpath, **kwargs)

    def finalize(self, **kwargs):
        self.set_title("{} Confusion Matrix".format(self.name))
        self.ax.set_ylabel("True Class")
        self.ax.set_xlabel("Predicted Class")

        # Call tight layout to maximize readability
        self.fig.tight_layout()


##########################################################################
## Quick Method
##########################################################################


def confusion_matrix(
    estimator,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    ax=None,
    sample_weight=None,
    percent=False,
    classes=None,
    encoder=None,
    cmap="YlOrRd",
    fontsize=None,
    is_fitted="auto",
    force_model=False,
    show=True,
    **kwargs
):
    """Confusion Matrix

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
    estimator : estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.

    X_train : array-like, 2D
        The table of instance data or independent variables that describe the outcome of
        the dependent variable, y. Used to fit the visualizer and also to score the
        visualizer if test splits are not specified.

    y_train : array-like, 2D
        The vector of target data or the dependent variable predicted by X. Used to fit
        the visualizer and also to score the visualizer if test splits are not
        specified.

    X_test: array-like, 2D, default: None
        The table of instance data or independent variables that describe the outcome of
        the dependent variable, y. Used to score the visualizer if specified.

    y_test: array-like, 1D, default: None
        The vector of target data or the dependent variable predicted by X. Used to
        score the visualizer if specified.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If not specified the current axes will be
        used (or generated if required).

    sample_weight: array-like of shape = [n_samples], optional
        Passed to ``confusion_matrix`` to weight the samples.

    percent: bool, default: False
        Determines whether or not the confusion_matrix is displayed as counts
        or as a percent of true predictions. Note, if specifying a subset of
        classes, percent should be set to False or inaccurate figures will be
        displayed.

    classes : list of str, defult: None
        The class labels to use for the legend ordered by the index of the sorted
        classes discovered in the ``fit()`` method. Specifying classes in this
        manner is used to change the class names to a more specific format or
        to label encoded integer classes. Some visualizers may also use this
        field to filter the visualization for specific classes. For more advanced
        usage specify an encoder rather than class labels.

    encoder : dict or LabelEncoder, default: None
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.

    cmap : string, default: ``'YlOrRd'``
        Specify a colormap to define the heatmap of the predicted class
        against the actual class in the confusion matrix.

    fontsize : int, default: None
        Specify the fontsize of the text in the grid and labels to make the
        matrix a bit easier to read. Uses rcParams font size by default.

    is_fitted : bool or str, default="auto"
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If "auto" (default), a helper method will check if the estimator
        is fitted before fitting it again.

    force_model : bool, default: False
        Do not check to ensure that the underlying estimator is a classifier. This
        will prevent an exception when the visualizer is initialized but may result
        in unexpected or unintended behavior.

    show: bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot
        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply
        calls ``finalize()``

    kwargs : dict
        Keyword arguments passed to the visualizer base classes.

    Returns
    -------
    viz : ConfusionMatrix
        Returns the fitted, finalized visualizer
    """
    # Instantiate the visualizer
    visualizer = ConfusionMatrix(
        estimator=estimator,
        ax=ax,
        sample_weight=sample_weight,
        percent=percent,
        classes=classes,
        encoder=encoder,
        cmap=cmap,
        fontsize=fontsize,
        is_fitted=is_fitted,
        force_model=force_model,
        **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X_train, y_train, **kwargs)

    # Scores the visualizer with X_test and y_test if provided,
    # X_train, y_train if not provided
    if X_test is not None and y_test is not None:
        visualizer.score(X_test, y_test)
    else:
        visualizer.score(X_train, y_train)

    if show:
        visualizer.show()
    else:
        visualizer.finalize()

    # Return the visualizer
    return visualizer
