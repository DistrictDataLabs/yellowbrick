# yellowbrick.classifier.rocauc
# Implements visual ROC/AUC curves for classification evaluation.
#
# Author:   Rebecca Bilbro
# Author:   Benjamin Bengfort
# Author:   Neal Humphrey
# Created:  Tue May 03 18:15:42 2017 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: rocauc.py [5388065] neal@nhumphrey.com $

"""
Implements visual ROC/AUC curves for classification evaluation.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import type_of_target

from yellowbrick.exceptions import ModelError
from yellowbrick.style.palettes import LINE_COLOR
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.classifier.base import ClassificationScoreVisualizer


# Dictionary keys for ROCAUC
MACRO = "macro"
MICRO = "micro"

# Target Type Constants
BINARY = "binary"
MULTICLASS = "multiclass"


##########################################################################
## ROCAUC Visualizer
##########################################################################


class ROCAUC(ClassificationScoreVisualizer):
    """
    Receiver Operating Characteristic (ROC) curves are a measure of a
    classifier's predictive quality that compares and visualizes the tradeoff
    between the models' sensitivity and specificity. The ROC curve displays
    the true positive rate on the Y axis and the false positive rate on the
    X axis on both a global average and per-class basis. The ideal point is
    therefore the top-left corner of the plot: false positives are zero and
    true positives are one.

    This leads to another metric, area under the curve (AUC), a computation
    of the relationship between false positives and true positives. The higher
    the AUC, the better the model generally is. However, it is also important
    to inspect the "steepness" of the curve, as this describes the
    maximization of the true positive rate while minimizing the false positive
    rate. Generalizing "steepness" usually leads to discussions about
    convexity, which we do not get into here.

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

    micro : bool, default: True
        Plot the micro-averages ROC curve, computed from the sum of all true
        positives and false positives across all classes. Micro is not defined
        for binary classification problems with estimators with only a
        decision_function method.

    macro : bool, default: True
        Plot the macro-averages ROC curve, which simply takes the average of
        curves across all classes. Macro is not defined for binary
        classification problems with estimators with only a decision_function
        method.

    per_class : bool, default: True
        Plot the ROC curves for each individual class. This should be set
        to false if only the macro or micro average curves are required. For true
        binary classifiers, setting per_class=False will plot the positive class
        ROC curve, and per_class=True will use ``1-P(1)`` to compute the curve of
        the negative class if only a decision_function method exists on the estimator.

    binary : bool, default: False
        This argument quickly resets the visualizer for true binary classification
        by updating the micro, macro, and per_class arguments to False (do not use
        in conjunction with those other arguments). Note that this is not a true
        hyperparameter to the visualizer, it just collects other parameters into
        a single, simpler argument.

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

    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting.

    score_ : float
        An evaluation metric of the classifier on test data produced when
        ``score()`` is called. This metric is between 0 and 1 -- higher scores are
        generally better. For classifiers, this score is usually accuracy, but
        if micro or macro is specified this returns an F1 score.

    target_type_ : string
        Specifies if the detected classification target was binary or multiclass.

    Notes
    -----
    ROC curves are typically used in binary classification, and in fact the
    Scikit-Learn ``roc_curve`` metric is only able to perform metrics for
    binary classifiers. As a result it is necessary to binarize the output or
    to use one-vs-rest or one-vs-all strategies of classification. The
    visualizer does its best to handle multiple situations, but exceptions can
    arise from unexpected models or outputs.

    Another important point is the relationship of class labels specified on
    initialization to those drawn on the curves. The classes are not used to
    constrain ordering or filter curves; the ROC computation happens on the
    unique values specified in the target vector to the ``score`` method. To
    ensure the best quality visualization, do not use a LabelEncoder for this
    and do not pass in class labels.

    .. seealso::
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    .. todo:: Allow the class list to filter the curves on the visualization.

    Examples
    --------
    >>> from yellowbrick.classifier import ROCAUC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_data("occupancy")
    >>> features = ["temp", "relative humidity", "light", "C02", "humidity"]
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> oz = ROCAUC(LogisticRegression())
    >>> oz.fit(X_train, y_train)
    >>> oz.score(X_test, y_test)
    >>> oz.show()
    """

    def __init__(
        self,
        estimator,
        ax=None,
        micro=True,
        macro=True,
        per_class=True,
        binary=False,
        classes=None,
        encoder=None,
        is_fitted="auto",
        force_model=False,
        **kwargs
    ):
        super(ROCAUC, self).__init__(
            estimator,
            ax=ax,
            classes=classes,
            encoder=encoder,
            is_fitted=is_fitted,
            force_model=force_model,
            **kwargs
        )

        # Set the visual parameters for ROCAUC
        # NOTE: the binary flag breaks our API since it's really just a meta parameter
        # for micro, macro, and per_class. We knew this going in, but did it anyway.
        if binary:
            self.micro = False
            self.macro = False
            self.per_class = False
        else:
            self.micro = micro
            self.macro = macro
            self.per_class = per_class

    def fit(self, X, y=None):
        """
        Fit the classification model.
        """
        # The target determines what kind of estimator is fit
        ttype = type_of_target(y)
        if ttype.startswith(MULTICLASS):
            self.target_type_ = MULTICLASS
        elif ttype.startswith(BINARY):
            self.target_type_ = BINARY
        else:
            raise YellowbrickValueError(
                (
                    "{} does not support target type '{}', "
                    "please provide a binary or multiclass single-output target"
                ).format(self.__class__.__name__, ttype)
            )

        # Fit the model and return self
        return super(ROCAUC, self).fit(X, y)

    def score(self, X, y=None):
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
        -------
        score_ : float
            Global accuracy unless micro or macro scores are requested.
        """
        # Call super to check if fitted and to compute self.score_
        # NOTE: this sets score to the base score if neither macro nor micro
        super(ROCAUC, self).score(X, y)

        # Compute the predictions for the test data
        y_pred = self._get_y_scores(X)

        if self.target_type_ == BINARY:
            # For binary, per_class must be True to draw micro/macro curves
            if (self.micro or self.macro) and not self.per_class:
                raise ModelError(
                    "no curves will be drawn; ",
                    "set per_class=True or micro=False and macro=False.",
                )

            # For binary, if predictions are returned in shape (n,), micro and macro
            # curves are not defined
            if (self.micro or self.macro) and len(y_pred.shape) == 1:
                raise ModelError(
                    "no curves will be drawn; set binary=True.",
                )

        if self.target_type_ == MULTICLASS:
            # If it's multiclass classification, at least one of micro, macro, or
            # per_class must be True
            if not self.micro and not self.macro and not self.per_class:
                raise YellowbrickValueError(
                    "no curves will be drawn; specify micro, macro, or per_class"
                )

        # Classes may be label encoded so only use what's in y to compute.
        # The self.classes_ attribute will be used as names for labels.
        classes = np.unique(y)
        n_classes = len(classes)

        # Store the false positive rate, true positive rate and curve info.
        self.fpr = dict()
        self.tpr = dict()
        self.roc_auc = dict()

        # If the decision is binary draw only ROC curve for the positive class
        if self.target_type_ is BINARY and not self.per_class:
            # In this case predict_proba returns an array of shape (n, 2) which
            # specifies the probabilities of both the negative and positive classes.
            if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
                self.fpr[BINARY], self.tpr[BINARY], _ = roc_curve(y, y_pred[:, 1])
            else:
                # decision_function returns array of shape (n,), so plot it directly
                self.fpr[BINARY], self.tpr[BINARY], _ = roc_curve(y, y_pred)
            self.roc_auc[BINARY] = auc(self.fpr[BINARY], self.tpr[BINARY])

        # Per-class binary decisions may have to have the negative class curve computed
        elif self.target_type_ is BINARY and self.per_class:
            # draw a curve for class 1 (the positive class)
            if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
                # predict_proba returns array of shape (n, 2), so use
                # probability of class 1 to compute ROC
                self.fpr[1], self.tpr[1], _ = roc_curve(y, y_pred[:, 1])
            else:
                # decision_function returns array of shape (n,)
                self.fpr[1], self.tpr[1], _ = roc_curve(y, y_pred)
            self.roc_auc[1] = auc(self.fpr[1], self.tpr[1])

            # draw a curve for class 0 (the negative class)
            if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
                # predict_proba returns array of shape (n, 2), so use
                # probability of class 0 to compute ROC
                self.fpr[0], self.tpr[0], _ = roc_curve(1 - y, y_pred[:, 0])
            else:
                # decision_function returns array of shape (n,).
                # To draw a ROC curve for class 0 we swap the classes 0 and 1 in y
                # and reverse classifiers predictions y_pred.
                self.fpr[0], self.tpr[0], _ = roc_curve(1 - y, -y_pred)
            self.roc_auc[0] = auc(self.fpr[0], self.tpr[0])

        else:
            # Otherwise compute the ROC curve and ROC area for each class
            for i, c in enumerate(classes):
                self.fpr[i], self.tpr[i], _ = roc_curve(y, y_pred[:, i], pos_label=c)
                self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])

        # Compute micro average
        if self.micro:
            self._score_micro_average(y, y_pred, classes, n_classes)

        # Compute macro average
        if self.macro:
            self._score_macro_average(n_classes)

        # Draw the Curves
        self.draw()

        # Set score to micro average if specified
        if self.micro:
            self.score_ = self.roc_auc[MICRO]

        # Set score to macro average if not micro
        if self.macro:
            self.score_ = self.roc_auc[MACRO]

        return self.score_

    def draw(self):
        """
        Renders ROC-AUC plot.
        Called internally by score, possibly more than once

        Returns
        -------
        ax : the axis with the plotted figure
        """
        colors = self.class_colors_[0 : len(self.classes_)]
        n_classes = len(colors)

        # If it's a binary decision, plot the single ROC curve
        if self.target_type_ == BINARY and not self.per_class:
            self.ax.plot(
                self.fpr[BINARY],
                self.tpr[BINARY],
                label="ROC for binary decision, AUC = {:0.2f}".format(
                    self.roc_auc[BINARY]
                ),
            )

        # If per-class plotting is requested, plot ROC curves for each class
        if self.per_class:
            for i, color in zip(range(n_classes), colors):
                self.ax.plot(
                    self.fpr[i],
                    self.tpr[i],
                    color=color,
                    label="ROC of class {}, AUC = {:0.2f}".format(
                        self.classes_[i], self.roc_auc[i]
                    ),
                )

        # If requested, plot the ROC curve for the micro average
        if self.micro:
            self.ax.plot(
                self.fpr[MICRO],
                self.tpr[MICRO],
                linestyle="--",
                color=self.class_colors_[len(self.classes_) - 1],
                label="micro-average ROC curve, AUC = {:0.2f}".format(
                    self.roc_auc["micro"]
                ),
            )

        # If requested, plot the ROC curve for the macro average
        if self.macro:
            self.ax.plot(
                self.fpr[MACRO],
                self.tpr[MACRO],
                linestyle="--",
                color=self.class_colors_[len(self.classes_) - 1],
                label="macro-average ROC curve, AUC = {:0.2f}".format(
                    self.roc_auc["macro"]
                ),
            )

        # Plot the line of no discrimination to compare the curve to.
        self.ax.plot([0, 1], [0, 1], linestyle=":", c=LINE_COLOR)
        return self.ax

    def finalize(self, **kwargs):
        """
        Sets a title and axis labels of the figures and ensures the axis limits
        are scaled between the valid ROCAUC score values.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        Notes
        -----
        Generally this method is called from show and not directly by the user.
        """
        # Set the title and add the legend
        self.set_title("ROC Curves for {}".format(self.name))
        self.ax.legend(loc="lower right", frameon=True)

        # Set the limits for the ROC/AUC (always between 0 and 1)
        self.ax.set_xlim([0.0, 1.0])
        self.ax.set_ylim([0.0, 1.0])

        # Set x and y axis labels
        self.ax.set_ylabel("True Positive Rate")
        self.ax.set_xlabel("False Positive Rate")

    def _get_y_scores(self, X):
        """
        The ``roc_curve`` metric requires target scores that can either be the
        probability estimates of the positive class, confidence values or non-
        thresholded measure of decisions (as returned by "decision_function").

        This method computes the scores by resolving the estimator methods
        that retreive these values.

        .. todo:: implement confidence values metric.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features -- generally the test data
            that is associated with y_true values.
        """
        # The resolution order of scoring functions
        attrs = ("predict_proba", "decision_function")

        # Return the first resolved function
        for attr in attrs:
            try:
                method = getattr(self.estimator, attr, None)
                if method:
                    return method(X)
            except AttributeError:
                # Some Scikit-Learn estimators have both probability and
                # decision functions but override __getattr__ and raise an
                # AttributeError on access.
                # Note that because of the ordering of our attrs above,
                # estimators with both will *only* ever use probability.
                continue

        # If we've gotten this far, raise an error
        raise ModelError(
            "ROCAUC requires estimators with predict_proba or "
            "decision_function methods."
        )

    def _score_micro_average(self, y, y_pred, classes, n_classes):
        """
        Compute the micro average scores for the ROCAUC curves.
        """
        # Convert y to binarized array for micro and macro scores
        y = label_binarize(y, classes=classes)
        if n_classes == 2:
            y = np.hstack((1 - y, y))

        # Compute micro-average
        self.fpr[MICRO], self.tpr[MICRO], _ = roc_curve(y.ravel(), y_pred.ravel())
        self.roc_auc[MICRO] = auc(self.fpr[MICRO], self.tpr[MICRO])

    def _score_macro_average(self, n_classes):
        """
        Compute the macro average scores for the ROCAUC curves.
        """
        # Gather all FPRs
        all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(n_classes)]))
        avg_tpr = np.zeros_like(all_fpr)

        # Compute the averages per class
        for i in range(n_classes):
            avg_tpr += np.interp(all_fpr, self.fpr[i], self.tpr[i])

        # Finalize the average
        avg_tpr /= n_classes

        # Store the macro averages
        self.fpr[MACRO] = all_fpr
        self.tpr[MACRO] = avg_tpr
        self.roc_auc[MACRO] = auc(self.fpr[MACRO], self.tpr[MACRO])


##########################################################################
## Quick method for ROCAUC
##########################################################################


def roc_auc(
    estimator,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    ax=None,
    micro=True,
    macro=True,
    per_class=True,
    binary=False,
    classes=None,
    encoder=None,
    is_fitted="auto",
    force_model=False,
    show=True,
    **kwargs
):
    """ROCAUC

    Receiver Operating Characteristic (ROC) curves are a measure of a
    classifier's predictive quality that compares and visualizes the tradeoff
    between the models' sensitivity and specificity. The ROC curve displays
    the true positive rate on the Y axis and the false positive rate on the
    X axis on both a global average and per-class basis. The ideal point is
    therefore the top-left corner of the plot: false positives are zero and
    true positives are one.

    This leads to another metric, area under the curve  (AUC), a computation
    of the relationship between false positives and true positives. The higher
    the AUC, the better the model generally is. However, it is also important
    to inspect the "steepness" of the curve, as this describes the
    maximization of the true positive rate while minimizing the false positive
    rate. Generalizing "steepness" usually leads to discussions about
    convexity, which we do not get into here.

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
        the visualizer and also to score the visualizer if test splits not specified.

    X_test: array-like, 2D, default: None
        The table of instance data or independent variables that describe the outcome of
        the dependent variable, y. Used to score the visualizer if specified.

    y_test: array-like, 1D, default: None
        The vector of target data or the dependent variable predicted by X.
        Used to score the visualizer if specified.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If not specified the current axes will be
        used (or generated if required).

    test_size : float, default=0.2
        The percentage of the data to reserve as test data.

    random_state : int or None, default=None
        The value to seed the random number generator for shuffling data.

    micro : bool, default: True
        Plot the micro-averages ROC curve, computed from the sum of all true
        positives and false positives across all classes. Micro is not defined
        for binary classification problems with estimators with only a
        decision_function method.

    macro : bool, default: True
        Plot the macro-averages ROC curve, which simply takes the average of
        curves across all classes. Macro is not defined for binary
        classification problems with estimators with only a decision_function
        method.

    per_class : bool, default: True
        Plot the ROC curves for each individual class. This should be set
        to false if only the macro or micro average curves are required. For true
        binary classifiers, setting per_class=False will plot the positive class
        ROC curve, and per_class=True will use ``1-P(1)`` to compute the curve of
        the negative class if only a decision_function method exists on the estimator.

    binary : bool, default: False
        This argument quickly resets the visualizer for true binary classification
        by updating the micro, macro, and per_class arguments to False (do not use
        in conjunction with those other arguments). Note that this is not a true
        hyperparameter to the visualizer, it just collects other parameters into
        a single, simpler argument.

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

    Notes
    -----
    ROC curves are typically used in binary classification, and in fact the
    Scikit-Learn ``roc_curve`` metric is only able to perform metrics for
    binary classifiers. As a result it is necessary to binarize the output or
    to use one-vs-rest or one-vs-all strategies of classification. The
    visualizer does its best to handle multiple situations, but exceptions can
    arise from unexpected models or outputs.

    Another important point is the relationship of class labels specified on
    initialization to those drawn on the curves. The classes are not used to
    constrain ordering or filter curves; the ROC computation happens on the
    unique values specified in the target vector to the ``score`` method. To
    ensure the best quality visualization, do not use a LabelEncoder for this
    and do not pass in class labels.

    .. seealso:: https://bit.ly/2IORWO2
    .. todo:: Allow the class list to filter the curves on the visualization.

    Examples
    --------
    >>> from yellowbrick.classifier import ROCAUC
    >>> from sklearn.linear_model import LogisticRegression
    >>> data = load_data("occupancy")
    >>> features = ["temp", "relative humidity", "light", "C02", "humidity"]
    >>> X = data[features].values
    >>> y = data.occupancy.values
    >>> roc_auc(LogisticRegression(), X, y)

    Returns
    -------
    viz : ROCAUC
        Returns the fitted, finalized visualizer object
    """
    # Instantiate the visualizer
    visualizer = ROCAUC(
        estimator=estimator,
        ax=ax,
        micro=micro,
        macro=macro,
        per_class=per_class,
        binary=binary,
        classes=classes,
        encoder=encoder,
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
