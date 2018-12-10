# yellowbrick.classifier.prcurve
# Implements Precision-Recall curves for classification models.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Tue Sep 04 16:47:19 2018 -0400
#
# ID: prcurve.py [] benjamin@bengfort.com $

"""
Implements Precision-Recall curves for classification models.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from ..exceptions import ModelError, NotFitted
from ..exceptions import YellowbrickValueError
from .base import ClassificationScoreVisualizer

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import precision_recall_curve as sk_precision_recall_curve


# Target Type Constants
BINARY = "binary"
MULTICLASS = "multiclass"

# Average Metric Constants
MICRO = "micro"


##########################################################################
## PrecisionRecallCurve Visualizer
##########################################################################

class PrecisionRecallCurve(ClassificationScoreVisualizer):
    """
    Precision-Recall curves are a metric used to evaluate a classifier's quality,
    particularly when classes are very imbalanced. The precision-recall curve
    shows the tradeoff between precision, a measure of result relevancy, and
    recall, a measure of how many relevant results are returned. A large area
    under the curve represents both high recall and precision, the best case
    scenario for a classifier, showing a model that returns accurate results
    for the majority of classes it selects.

    .. todo:: extend docstring

    Parameters
    ----------
    model : the Scikit-Learn estimator
        A classification model to score the precision-recall curve on.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    classes : list
        A list of class names for the legend. If classes is None and a y value
        is passed to fit then the classes are selected from the target vector.
        Note that the curves must be computed based on what is in the target
        vector passed to the ``score()`` method. Class names are used for
        labeling only and must be in the correct order to prevent confusion.

    fill_area : bool, default=True
        Fill the area under the curve (or curves) with the curve color.

    ap_score : bool, default=True
        Annotate the graph with the average precision score, a summary of the
        plot that is computed as the weighted mean of precisions at each
        threshold, with the increase in recall from the previous threshold used
        as the weight.

    micro : bool, default=True
        If multi-class classification, draw the precision-recall curve for the
        micro-average of all classes. In the multi-class case, either micro or
        per-class must be set to True. Ignored in the binary case.

    iso_f1_curves : bool, default=False
        Draw ISO F1-Curves on the plot to show how close the precision-recall
        curves are to different F1 scores.

    per_class : bool, default=False
        If multi-class classification, draw the precision-recall curve for
        each class using a OneVsRestClassifier to compute the recall on a
        per-class basis. In the multi-class case, either micro or per-class
        must be set to True. Ignored in the binary case.

    fill_opacity : float, default=0.2
        Specify the alpha or opacity of the fill area (0 being transparent,
        and 1.0 being completly opaque).

    line_opacity : float, default=0.8
        Specify the alpha or opacity of the lines (0 being transparent, and
        1.0 being completly opaque).

    kwargs : dict
        Keyword arguments passed to the visualization base class.

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


    Example
    -------
    >>> from yellowbrick.classifier import PrecisionRecallCurve
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.svm import LinearSVC
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> viz = PrecisionRecallCurve(LinearSVC())
    >>> viz.fit(X_train, y_train)
    >>> viz.score(X_test, y_test)
    >>> viz.poof()

    Notes
    -----

    .. seealso:: http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    """

    def __init__(self, model, ax=None, classes=None, fill_area=True, ap_score=True,
                 micro=True, iso_f1_curves=False, per_class=False, fill_opacity=0.2,
                 line_opacity=0.8, **kwargs):
        super(PrecisionRecallCurve, self).__init__(model, ax=ax, classes=classes, **kwargs)

        # Set visual params
        self.set_params(
            fill_area=fill_area,
            ap_score=ap_score,
            micro=micro,
            iso_f1_curves=iso_f1_curves,
            per_class=per_class,
            fill_opacity=fill_opacity,
            line_opacity=line_opacity,
        )

    def fit(self, X, y=None):
        """
        Fit the classification model; if y is multi-class, then the estimator
        is adapted with a OneVsRestClassifier strategy, otherwise the estimator
        is fit directly.
        """
        # The target determines what kind of estimator is fit
        ttype = type_of_target(y)
        if ttype.startswith(MULTICLASS):
            self.target_type_ = MULTICLASS
            self.estimator = OneVsRestClassifier(self.estimator)

            # Use label_binarize to create multi-label ouptut for OneVsRestClassifier
            Y = label_binarize(y, classes=np.unique(y))
        elif ttype.startswith(BINARY):
            self.target_type_ = BINARY

            # Different variable is used here to prevent transformation
            Y = y
        else:
            raise YellowbrickValueError((
                "{} does not support target type '{}', "
                "please provide a binary or multiclass single-output target"
            ).format(
                self.__class__.__name__, ttype
            ))

        # Fit the model and return self
        return super(PrecisionRecallCurve, self).fit(X, Y)

    def score(self, X, y=None):
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
            raise NotFitted((
                "{} cannot wrap an already fitted estimator"
            ).format(
                self.__class__.__name__
            ))

        # Compute the prediction/threshold scores
        y_scores = self._get_y_scores(X)

        # Handle binary and multiclass cases to create correct data structure
        if self.target_type_ == BINARY:
            self.precision_, self.recall_, _ = sk_precision_recall_curve(y, y_scores)
            self.score_ = average_precision_score(y, y_scores)
        else:
            # Use label_binarize to create multi-label ouptut for OneVsRestClassifier
            Y = label_binarize(y, classes=self.classes_)

            self.precision_, self.recall_, self.score_ = {}, {}, {}

            # Compute PRCurve for all classes
            for i, class_i in enumerate(self.classes_):
                self.precision_[class_i], self.recall_[class_i], _ = sk_precision_recall_curve(Y[:,i], y_scores[:,i])
                self.score_[class_i] = average_precision_score(Y[:,i], y_scores[:,i])

            # Compute micro average PR curve
            self.precision_[MICRO], self.recall_[MICRO], _ = sk_precision_recall_curve(
                Y.ravel(), y_scores.ravel()
            )
            self.score_[MICRO] = average_precision_score(Y, y_scores, average=MICRO)

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
        if self.iso_f1_curves:
            for f1 in np.linspace(0.2, 0.8, num=4):
                x = np.linspace(0.01, 1)
                y = f1 * x / (2 * x - f1)
                self.ax.plot(x[y>=0], y[y>=0], color='#333333', alpha=0.2)
                self.ax.annotate('$f_1={:0.1f}$'.format(f1), xy=(0.9, y[45]+0.02))

        if self.target_type_ == BINARY:
            return self._draw_binary()
        return self._draw_multiclass()

    def _draw_binary(self):
        """
        Draw the precision-recall curves in the binary case
        """
        self._draw_pr_curve(self.recall_, self.precision_, label="binary PR curve")
        self._draw_ap_score(self.score_)


    def _draw_multiclass(self):
        """
        Draw the precision-recall curves in the multiclass case
        """
        # TODO: handle colors better with a mapping and user input
        if self.per_class:
            for cls in self.classes_:
                precision = self.precision_[cls]
                recall = self.recall_[cls]

                label = "PR for class {} (area={:0.2f})".format(cls, self.score_[cls])
                self._draw_pr_curve(recall, precision, label=label)

        if self.micro:
            precision = self.precision_[MICRO]
            recall = self.recall_[MICRO]
            self._draw_pr_curve(recall, precision)

        self._draw_ap_score(self.score_[MICRO])

    def _draw_pr_curve(self, recall, precision, label=None):
        """
        Helper function to draw a precision-recall curve with specified settings
        """
        self.ax.step(
            recall, precision, alpha=self.line_opacity, where='post', label=label
        )
        if self.fill_area:
            self.ax.fill_between(recall, precision, step='post', alpha=self.fill_opacity)

    def _draw_ap_score(self, score, label=None):
        """
        Helper function to draw the AP score annotation
        """
        label = label or "Avg Precision={:0.2f}".format(score)
        if self.ap_score:
            self.ax.axhline(
                y=score, color="r", ls="--", label=label
            )

    def finalize(self):
        """
        Finalize the figure by adding titles, labels, and limits.
        """
        self.set_title("Precision-Recall Curve for {}".format(self.name))
        self.ax.legend(loc='lower left', frameon=True)

        self.ax.set_xlim([0.0, 1.0])
        self.ax.set_ylim([0.0, 1.0])

        self.ax.set_ylabel("Precision")
        self.ax.set_xlabel("Recall")

    def _get_y_scores(self, X):
        """
        The ``precision_recall_curve`` metric requires target scores that
        can either be the probability estimates of the positive class,
        confidence values, or non-thresholded measures of decisions (as
        returned by a "decision function").
        """
        # TODO refactor shared method with ROCAUC

        # Resolution order of scoring functions
        attrs = (
            'decision_function',
            'predict_proba',
        )

        # Return the first resolved function
        for attr in attrs:
            try:
                method = getattr(self.estimator, attr, None)
                if method:
                    # Compute the scores from the decision function
                    y_scores = method(X)

                    # Return only the positive class for binary predict_proba
                    if self.target_type_ == BINARY and y_scores.ndim == 2:
                        return y_scores[:,1]
                    return y_scores

            except AttributeError:
                # Some Scikit-Learn estimators have both probability and
                # decision functions but override __getattr__ and raise an
                # AttributeError on access.
                continue

        # If we've gotten this far, we can't do anything
        raise ModelError((
            "{} requires estimators with predict_proba or decision_function methods."
        ).format(self.__class__.__name__))


# Alias for PrecisionRecallCurve
PRCurve = PrecisionRecallCurve


##########################################################################
## Quick Method
##########################################################################

def precision_recall_curve(model, X, y, ax=None, train_size=0.8,
                           random_state=None, shuffle=True, **kwargs):
    """Precision-Recall Curve quick method:

    Parameters
    ----------
    model : the Scikit-Learn estimator
        A classification model to score the precision-recall curve on.

    X : ndarray or DataFrame of shape n x m
        A feature array of n instances with m features the model is trained on.
        This array will be split into train and test splits.

    y : ndarray or Series of length n
        An array or series of target or class values. This vector will be split
        into train and test splits.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    train_size : float or int, default=0.8
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the train split. If int, represents the
        absolute number of train samples.

    random_state : int, RandomState, or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.

    classes : list
        A list of class names for the legend. If classes is None and a y value
        is passed to fit then the classes are selected from the target vector.
        Note that the curves must be computed based on what is in the target
        vector passed to the ``score()`` method. Class names are used for
        labeling only and must be in the correct order to prevent confusion.

    fill_area : bool, default=True
        Fill the area under the curve (or curves) with the curve color.

    ap_score : bool, default=True
        Annotate the graph with the average precision score, a summary of the
        plot that is computed as the weighted mean of precisions at each
        threshold, with the increase in recall from the previous threshold used
        as the weight.

    micro : bool, default=True
        If multi-class classification, draw the precision-recall curve for the
        micro-average of all classes. In the multi-class case, either micro or
        per-class must be set to True. Ignored in the binary case.

    iso_f1_curves : bool, default=False
        Draw ISO F1-Curves on the plot to show how close the precision-recall
        curves are to different F1 scores.

    per_class : bool, default=False
        If multi-class classification, draw the precision-recall curve for
        each class using a OneVsRestClassifier to compute the recall on a
        per-class basis. In the multi-class case, either micro or per-class
        must be set to True. Ignored in the binary case.

    fill_opacity : float, default=0.2
        Specify the alpha or opacity of the fill area (0 being transparent,
        and 1.0 being completly opaque).

    line_opacity : float, default=0.8
        Specify the alpha or opacity of the lines (0 being transparent, and
        1.0 being completly opaque).

    kwargs : dict
        Keyword arguments passed to the visualization base class.

    Returns
    -------
    viz : PrecisionRecallCurve
        Returns the visualizer that generates the curve visualization.

    Notes
    -----
    Data is split using ``sklearn.model_selection.train_test_split`` before
    computing the Precision-Recall curve. Splitting options such as train_size,
    random_state, and shuffle are specified. Note that splits are not stratified,
    if required, it is recommended to use the base class.
    """
    # Instantiate the visualizer
    viz = PRCurve(model, ax=ax, **kwargs)

    # Create train and test splits to validate the model
    X_train, X_test, y_train, y_test = tts(
        X, y, train_size=train_size, random_state=random_state, shuffle=shuffle
    )

    # Fit and transform the visualizer
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.finalize()

    # Return the visualizer
    return viz
