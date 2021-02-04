# yellowbrick.classifier.prcurve
# Implements Precision-Recall curves for classification models.
#
# Author:  Benjamin Bengfort
# Created: Tue Sep 04 16:47:19 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: prcurve.py [48889c4] benjamin@bengfort.com $

"""
Implements Precision-Recall curves for classification models.
"""

##########################################################################
## Imports
##########################################################################

import warnings
import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve as sk_precision_recall_curve

from yellowbrick.style.colors import resolve_colors
from yellowbrick.exceptions import YellowbrickWarning
from yellowbrick.exceptions import ModelError, NotFitted
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.classifier.base import ClassificationScoreVisualizer


# Target Type Constants
# TODO: These can now be imported from utils.target
BINARY = "binary"
MULTICLASS = "multiclass"

# Average Metric Constants
MICRO = "micro"

# Default Values
DEFAULT_ISO_F1_VALUES = (0.2, 0.4, 0.6, 0.8)


##########################################################################
## PrecisionRecallCurve Visualizer
##########################################################################


class PrecisionRecallCurve(ClassificationScoreVisualizer):
    """
    Precision-Recall curves are a metric used to evaluate a classifier's quality,
    particularly when classes are very imbalanced. The precision-recall curve
    shows the tradeoff between precision, a measure of result relevancy, and
    recall, a measure of completeness. For each class, precision is defined as
    the ratio of true positives to the sum of true and false positives, and
    recall is the ratio of true positives to the sum of true positives and false
    negatives.

    A large area under the curve represents both high recall and precision, the
    best case scenario for a classifier, showing a model that returns accurate
    results for the majority of classes it selects.

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

    classes : list of str, default: None
        The class labels to use for the legend ordered by the index of the sorted
        classes discovered in the ``fit()`` method. Specifying classes in this
        manner is used to change the class names to a more specific format or
        to label encoded integer classes. Some visualizers may also use this
        field to filter the visualization for specific classes. For more advanced
        usage specify an encoder rather than class labels.

    colors : list of strings,  default: None
        An optional list or tuple of colors to colorize the curves when
        ``per_class=True``. If ``per_class=False``, this parameter will
        be ignored. If both ``colors`` and ``cmap`` are provided,
        ``cmap`` will be ignored.

    cmap : string or Matplotlib colormap, default: None
        An optional string or Matplotlib colormap to colorize the curves
        when ``per_class=True``. If ``per_class=False``, this parameter
        will be ignored. If both ``colors`` and ``cmap`` are provided,
        ``cmap`` will be ignored.

    encoder : dict or LabelEncoder, default: None
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.

    fill_area : bool, default: True
        Fill the area under the curve (or curves) with the curve color.

    ap_score : bool, default: True
        Annotate the graph with the average precision score, a summary of the
        plot that is computed as the weighted mean of precisions at each
        threshold, with the increase in recall from the previous threshold used
        as the weight.

    micro : bool, default: True
        If multi-class classification, draw the precision-recall curve for the
        micro-average of all classes. In the multi-class case, either micro or
        per-class must be set to True. Ignored in the binary case.

    iso_f1_curves : bool, default: False
        Draw ISO F1-Curves on the plot to show how close the precision-recall
        curves are to different F1 scores.

    iso_f1_values : tuple , default: (0.2, 0.4, 0.6, 0.8)
        Values of f1 score for which to draw ISO F1-Curves

    per_class : bool, default: False
        If multi-class classification, draw the precision-recall curve for
        each class using a OneVsRestClassifier to compute the recall on a
        per-class basis. In the multi-class case, either micro or per-class
        must be set to True. Ignored in the binary case.

    fill_opacity : float, default: 0.2
        Specify the alpha or opacity of the fill area (0 being transparent,
        and 1.0 being completly opaque).

    line_opacity : float, default: 0.8
        Specify the alpha or opacity of the lines (0 being transparent, and
        1.0 being completly opaque).

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
    target_type_ : str
        Either ``"binary"`` or ``"multiclass"`` depending on the type of target
        fit to the visualizer. If ``"multiclass"`` then the estimator is
        wrapped in a OneVsRestClassifier classification strategy.

    score_ : float or dict of floats
        Average precision, a summary of the plot as a weighted mean of
        precision at each threshold, weighted by the increase in recall from
        the previous threshold. In the multiclass case, a mapping of class/metric
        to the average precision score.

    precision_ : array or dict of array with shape=[n_thresholds + 1]
        Precision values such that element i is the precision of predictions
        with score >= thresholds[i] and the last element is 1. In the multiclass
        case, a mapping of class/metric to precision array.

    recall_ : array or dict of array with shape=[n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
        In the multiclass case, a mapping of class/metric to recall array.

    classes_ : ndarray of shape (n_classes,)
        The class labels observed while fitting.

    class_count_ : ndarray of shape (n_classes,)
        Number of samples encountered for each class during fitting.


    Examples
    --------

    >>> from yellowbrick.classifier import PrecisionRecallCurve
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.svm import LinearSVC
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> viz = PrecisionRecallCurve(LinearSVC())
    >>> viz.fit(X_train, y_train)
    >>> viz.score(X_test, y_test)
    >>> viz.show()

    Notes
    -----
    To support multi-label classification, the estimator is wrapped in a
    ``OneVsRestClassifier`` to produce binary comparisons for each class
    (e.g. the positive case is the class and the negative case is any other
    class). The precision-recall curve can then be computed as the micro-average
    of the precision and recall for all classes (by setting micro=True), or individual
    curves can be plotted for each class (by setting per_class=True).

    Note also that some parameters of this visualizer are learned on the ``score``
    method, not only on ``fit``.

    .. seealso:: https://bit.ly/2kOIeCC
    """

    def __init__(
        self,
        estimator,
        ax=None,
        classes=None,
        colors=None,
        cmap=None,
        encoder=None,
        fill_area=True,
        ap_score=True,
        micro=True,
        iso_f1_curves=False,
        iso_f1_values=DEFAULT_ISO_F1_VALUES,
        per_class=False,
        fill_opacity=0.2,
        line_opacity=0.8,
        is_fitted="auto",
        force_model=False,
        **kwargs
    ):
        super(PrecisionRecallCurve, self).__init__(
            estimator,
            ax=ax,
            classes=classes,
            encoder=encoder,
            is_fitted=is_fitted,
            force_model=force_model,
            **kwargs
        )

        # Set visual params
        self.fill_area = fill_area
        self.ap_score = ap_score
        self.colors = colors
        self.cmap = cmap
        self.micro = micro
        self.iso_f1_curves = iso_f1_curves
        self.iso_f1_values = set(iso_f1_values)
        self.per_class = per_class
        self.fill_opacity = fill_opacity
        self.line_opacity = line_opacity

        if self.micro and self.per_class:
            warnings.warn(
                "micro=True is ignored;"
                "specify per_class=False to draw a PR curve after micro-averaging",
                YellowbrickWarning,
            )

    def fit(self, X, y=None):
        """
        Fit the classification model; if ``y`` is multi-class, then the estimator
        is adapted with a ``OneVsRestClassifier`` strategy, otherwise the estimator
        is fit directly.
        """
        # The target determines what kind of estimator is fit
        ttype = type_of_target(y)
        self._target_labels = np.unique(y)
        if ttype.startswith(MULTICLASS):
            self.target_type_ = MULTICLASS
            self.estimator = OneVsRestClassifier(self.estimator)

            # Use label_binarize to create multi-label output for OneVsRestClassifier
            Y = label_binarize(y, classes=self._target_labels)
        elif ttype.startswith(BINARY):
            # Different variable is used here to prevent transformation
            Y = y
            self.target_type_ = BINARY
        else:
            raise YellowbrickValueError(
                (
                    "{} does not support target type '{}', "
                    "please provide a binary or multiclass single-output target"
                ).format(self.__class__.__name__, ttype)
            )

        # Fit the model and return self
        return super(PrecisionRecallCurve, self).fit(X, Y)

    def score(self, X, y):
        """
        Generates the Precision-Recall curve on the specified test data.

        Returns
        -------
        score_ : float
            Average precision, a summary of the plot as a weighted mean of
            precision at each threshold, weighted by the increase in recall from
            the previous threshold.

        """
        # If we don't do this check, then it is possible that OneVsRestClassifier
        # has not correctly been fitted for multi-class targets.
        if not hasattr(self, "target_type_"):
            raise NotFitted.from_estimator(self, "score")

        # Must perform label binarization before calling super
        if self.target_type_ == MULTICLASS:
            # Use label_binarize to create multi-label output for OneVsRestClassifier
            y = label_binarize(y, classes=self._target_labels)

        # Call super to check if fitted and to compute classes_
        # Note that self.score_ computed in super will be overridden below
        super(PrecisionRecallCurve, self).score(X, y)

        # Compute the prediction/threshold scores
        y_scores = self._get_y_scores(X)

        # Handle binary and multiclass cases to create correct data structure
        if self.target_type_ == BINARY:
            self.precision_, self.recall_, _ = sk_precision_recall_curve(y, y_scores)
            self.score_ = average_precision_score(y, y_scores)
        else:
            self.precision_, self.recall_, self.score_ = {}, {}, {}

            # Compute PRCurve for all classes
            for i, class_i in enumerate(self.classes_):
                self.precision_[class_i], self.recall_[
                    class_i
                ], _ = sk_precision_recall_curve(y[:, i], y_scores[:, i])
                self.score_[class_i] = average_precision_score(y[:, i], y_scores[:, i])

            # Compute micro average PR curve
            self.precision_[MICRO], self.recall_[MICRO], _ = sk_precision_recall_curve(
                y.ravel(), y_scores.ravel()
            )
            self.score_[MICRO] = average_precision_score(y, y_scores, average=MICRO)

        # Draw the figure
        self.draw()

        # Return a score between 0 and 1
        if self.target_type_ == BINARY:
            return self.score_
        return self.score_[MICRO]

    def draw(self):
        """
        Draws the precision-recall curves computed in score on the axes.
        """
        # set the colors
        self._colors = resolve_colors(
            n_colors=len(self.classes_), colormap=self.cmap, colors=self.colors
        )

        if self.iso_f1_curves:
            for f1 in self.iso_f1_values:
                x = np.linspace(0.01, 1)
                y = f1 * x / (2 * x - f1)
                self.ax.plot(x[y >= 0], y[y >= 0], color="#333333", alpha=0.2)
                self.ax.annotate("$f_1={:0.1f}$".format(f1), xy=(0.9, y[45] + 0.02))

        if self.target_type_ == BINARY:
            return self._draw_binary()
        return self._draw_multiclass()

    def _draw_binary(self):
        """
        Draw the precision-recall curves in the binary case
        """
        self._draw_pr_curve(self.recall_, self.precision_, label="Binary PR curve")
        self._draw_ap_score(self.score_)

    def _draw_multiclass(self):
        """
        Draw the precision-recall curves in the multiclass case
        """
        if self.per_class:

            colors = dict(zip(self.classes_, self._colors))

            for cls in self.classes_:
                precision = self.precision_[cls]
                recall = self.recall_[cls]

                label = "PR for class {} (area={:0.2f})".format(cls, self.score_[cls])
                self._draw_pr_curve(recall, precision, label=label, color=colors[cls])

        elif self.micro:
            precision = self.precision_[MICRO]
            recall = self.recall_[MICRO]
            label = "Micro-average PR for all classes"
            self._draw_pr_curve(recall, precision, label=label)

        self._draw_ap_score(self.score_[MICRO])

    def _draw_pr_curve(self, recall, precision, label=None, color=None):
        """
        Helper function to draw a precision-recall curve with specified settings
        """
        self.ax.step(
            recall,
            precision,
            alpha=self.line_opacity,
            where="post",
            label=label,
            color=color,
        )
        if self.fill_area and not self.per_class:
            self.ax.fill_between(
                recall, precision, step="post", alpha=self.fill_opacity, color=color
            )

    def _draw_ap_score(self, score, label=None):
        """
        Helper function to draw the AP score annotation
        """
        label = label or "Avg. precision={:0.2f}".format(score)
        if self.ap_score:
            self.ax.axhline(y=score, color="r", ls="--", label=label)

    def finalize(self):
        """
        Finalize the figure by adding titles, labels, and limits.
        """
        self.set_title("Precision-Recall Curve for {}".format(self.name))
        self.ax.legend(loc="lower left", frameon=True)

        self.ax.set_xlim([0.0, 1.0])
        self.ax.set_ylim([0.0, 1.0])

        self.ax.set_ylabel("Precision")
        self.ax.set_xlabel("Recall")

        self.ax.grid(False)

    def _get_y_scores(self, X):
        """
        The ``precision_recall_curve`` metric requires target scores that
        can either be the probability estimates of the positive class,
        confidence values, or non-thresholded measures of decisions (as
        returned by a "decision function").
        """
        # TODO refactor shared method with ROCAUC

        # Resolution order of scoring functions
        attrs = ("decision_function", "predict_proba")

        # Return the first resolved function
        for attr in attrs:
            try:
                method = getattr(self.estimator, attr, None)
                if method:
                    # Compute the scores from the decision function
                    y_scores = method(X)

                    # Return only the positive class for binary predict_proba
                    if self.target_type_ == BINARY and y_scores.ndim == 2:
                        return y_scores[:, 1]
                    return y_scores

            except AttributeError:
                # Some Scikit-Learn estimators have both probability and
                # decision functions but override __getattr__ and raise an
                # AttributeError on access.
                continue

        # If we've gotten this far, we can't do anything
        raise ModelError(
            (
                "{} requires an estimator with predict_proba or decision_function."
            ).format(self.__class__.__name__)
        )


# Alias for PrecisionRecallCurve
PRCurve = PrecisionRecallCurve


##########################################################################
## Quick Method
##########################################################################


def precision_recall_curve(
    estimator,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    ax=None,
    classes=None,
    colors=None,
    cmap=None,
    encoder=None,
    fill_area=True,
    ap_score=True,
    micro=True,
    iso_f1_curves=False,
    iso_f1_values=DEFAULT_ISO_F1_VALUES,
    per_class=False,
    fill_opacity=0.2,
    line_opacity=0.8,
    is_fitted="auto",
    force_model=False,
    show=True,
    **kwargs
):
    """Precision-Recall Curve

    Precision-Recall curves are a metric used to evaluate a classifier's quality,
    particularly when classes are very imbalanced. The precision-recall curve
    shows the tradeoff between precision, a measure of result relevancy, and
    recall, a measure of completeness. For each class, precision is defined as
    the ratio of true positives to the sum of true and false positives, and
    recall is the ratio of true positives to the sum of true positives and false
    negatives.

    A large area under the curve represents both high recall and precision, the
    best case scenario for a classifier, showing a model that returns accurate
    results for the majority of classes it selects.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.

    X_train : ndarray or DataFrame of shape n x m
        A feature array of n instances with m features the model is trained on.
        Used to fit the visualizer and also to score the visualizer if test splits are
        not directly specified.

    y_train : ndarray or Series of length n
        An array or series of target or class values. Used to fit the visualizer and
        also to score the visualizer if test splits are not specified.

    X_test : ndarray or DataFrame of shape n x m, default: None
        An optional feature array of n instances with m features that the model
        is scored on if specified, using X_train as the training data.

    y_test : ndarray or Series of length n, default: None
        An optional array or series of target or class values that serve as actual
        labels for X_test for scoring purposes.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If not specified the current axes will be
        used (or generated if required).

    classes : list of str, default: None
        The class labels to use for the legend ordered by the index of the sorted
        classes discovered in the ``fit()`` method. Specifying classes in this
        manner is used to change the class names to a more specific format or
        to label encoded integer classes. Some visualizers may also use this
        field to filter the visualization for specific classes. For more advanced
        usage specify an encoder rather than class labels.

    colors : list of strings,  default: None
        An optional list or tuple of colors to colorize the curves when
        ``per_class=True``. If ``per_class=False``, this parameter will
        be ignored. If both ``colors`` and ``cmap`` are provided,
        ``cmap`` will be ignored.

    cmap : string or Matplotlib colormap, default: None
        An optional string or Matplotlib colormap to colorize the curves
        when ``per_class=True``. If ``per_class=False``, this parameter
        will be ignored. If both ``colors`` and ``cmap`` are provided,
        ``cmap`` will be ignored.

    encoder : dict or LabelEncoder, default: None
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.

    fill_area : bool, default: True
        Fill the area under the curve (or curves) with the curve color.

    ap_score : bool, default: True
        Annotate the graph with the average precision score, a summary of the
        plot that is computed as the weighted mean of precisions at each
        threshold, with the increase in recall from the previous threshold used
        as the weight.

    micro : bool, default: True
        If multi-class classification, draw the precision-recall curve for the
        micro-average of all classes. In the multi-class case, either micro or
        per-class must be set to True. Ignored in the binary case.

    iso_f1_curves : bool, default: False
        Draw ISO F1-Curves on the plot to show how close the precision-recall
        curves are to different F1 scores.

    iso_f1_values : tuple , default: (0.2, 0.4, 0.6, 0.8)
        Values of f1 score for which to draw ISO F1-Curves

    per_class : bool, default: False
        If multi-class classification, draw the precision-recall curve for
        each class using a OneVsRestClassifier to compute the recall on a
        per-class basis. In the multi-class case, either micro or per-class
        must be set to True. Ignored in the binary case.

    fill_opacity : float, default: 0.2
        Specify the alpha or opacity of the fill area (0 being transparent,
        and 1.0 being completly opaque).

    line_opacity : float, default: 0.8
        Specify the alpha or opacity of the lines (0 being transparent, and
        1.0 being completly opaque).

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
    viz : PrecisionRecallCurve
        Returns the visualizer that generates the curve visualization.
    """
    # Instantiate the visualizer
    viz = PRCurve(
        estimator,
        ax=ax,
        classes=classes,
        colors=colors,
        cmap=cmap,
        encoder=encoder,
        fill_area=fill_area,
        ap_score=ap_score,
        micro=micro,
        iso_f1_curves=iso_f1_curves,
        iso_f1_values=iso_f1_values,
        per_class=per_class,
        fill_opacity=fill_opacity,
        line_opacity=line_opacity,
        is_fitted=is_fitted,
        force_model=force_model,
        **kwargs
    )

    # Fit the visualizer
    viz.fit(X_train, y_train)

    # Score the visualizer
    if X_test is not None and y_test is not None:
        viz.score(X_test, y_test)
    elif X_test is not None or y_test is not None:
        raise YellowbrickValueError(
            "both X_test and y_test are required if one is specified"
        )
    else:
        viz.score(X_train, y_train)

    # Draw the final visualization
    if show:
        viz.show()
    else:
        viz.finalize()

    # Return the visualizer
    return viz
