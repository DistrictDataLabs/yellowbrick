# yellowbrick.classifier.calibration
# Calibration curve (reliability diagram) for binary classifiers.
"""
Calibration curve visualizer: do predicted probabilities match reality?
"""

import numpy as np
from sklearn.calibration import calibration_curve as sk_calibration_curve
from sklearn.metrics import brier_score_loss

from yellowbrick.classifier.base import ClassificationScoreVisualizer
from yellowbrick.exceptions import ModelError, YellowbrickValueError

__all__ = ["CalibrationCurve", "calibration_curve"]

STRATEGIES = ("uniform", "quantile")


class CalibrationCurve(ClassificationScoreVisualizer):
    """
    Plot the fraction of true positives against the mean predicted
    probability for each bin (a reliability diagram) and report the
    Brier score. The closer the curve is to the diagonal, the better
    calibrated the model is.

    Parameters
    ----------
    estimator : estimator
        A binary classifier that implements ``predict_proba``.
    ax : matplotlib Axes, default: None
        The axes to plot on. If None, the current axes are used.
    n_bins : int, default: 10
        Number of bins used to discretize the predicted probabilities.
    strategy : {"uniform", "quantile"}, default: "uniform"
        Equal-width bins ("uniform") or equal-count bins ("quantile").
    kwargs : dict
        Passed to the base visualizer.

    Attributes
    ----------
    prob_true_ : ndarray
        Fraction of positives in each bin.
    prob_pred_ : ndarray
        Mean predicted probability in each bin.
    brier_score_ : float
        Brier score loss (lower is better).

    Examples
    --------
    >>> viz = CalibrationCurve(LogisticRegression())
    >>> viz.fit(X_train, y_train)
    >>> viz.score(X_test, y_test)
    >>> viz.show()

    Notes
    -----
    Only binary classification is supported. The positive class is the
    second entry of ``estimator.classes_``.
    """

    def __init__(self, estimator, ax=None, n_bins=10, strategy="uniform", **kwargs):
        if strategy not in STRATEGIES:
            raise YellowbrickValueError(
                "strategy must be one of {}, got '{}'".format(STRATEGIES, strategy)
            )
        if int(n_bins) < 1:
            raise YellowbrickValueError("n_bins must be a positive integer")
        super(CalibrationCurve, self).__init__(estimator, ax=ax, **kwargs)
        self.n_bins = int(n_bins)
        self.strategy = strategy

    def score(self, X, y):
        """Compute the calibration curve and Brier score on (X, y)."""
        if not hasattr(self.estimator, "predict_proba"):
            raise ModelError("CalibrationCurve requires an estimator with predict_proba")

        y = np.asarray(y)
        classes = self.estimator.classes_
        if len(classes) != 2:
            raise YellowbrickValueError("CalibrationCurve supports binary classification only")

        y_true = (y == classes[1]).astype(int)
        proba = self.estimator.predict_proba(X)[:, 1]

        self.prob_true_, self.prob_pred_ = sk_calibration_curve(
            y_true, proba, n_bins=self.n_bins, strategy=self.strategy
        )
        self.brier_score_ = brier_score_loss(y_true, proba)
        self.score_ = self.brier_score_

        self.draw()
        return self.score_

    def draw(self):
        """Draw the reference diagonal and the calibration curve."""
        self.ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        self.ax.plot(
            self.prob_pred_,
            self.prob_true_,
            "s-",
            label="{} (Brier={:.3f})".format(self.name, self.brier_score_),
        )
        return self.ax

    def finalize(self, **kwargs):
        """Add title, axis labels, limits and legend."""
        self.set_title("Calibration Curve for {}".format(self.name))
        self.ax.set_xlabel("Mean predicted probability")
        self.ax.set_ylabel("Fraction of positives")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1.05)
        self.ax.legend(loc="upper left")


def calibration_curve(
    estimator, X_train, y_train, X_test=None, y_test=None,
    ax=None, n_bins=10, strategy="uniform", show=True, **kwargs
):
    """
    Quick method: fit a CalibrationCurve on the training data, score it
    on the test data (or the training data if no test set is given) and
    return the visualizer.
    """
    viz = CalibrationCurve(estimator, ax=ax, n_bins=n_bins, strategy=strategy, **kwargs)
    viz.fit(X_train, y_train)
    if X_test is not None and y_test is not None:
        viz.score(X_test, y_test)
    else:
        viz.score(X_train, y_train)

    if show:
        viz.show()
    else:
        viz.finalize()
    return viz
