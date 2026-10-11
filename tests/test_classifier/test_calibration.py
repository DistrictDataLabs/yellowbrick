# tests.test_classifier.test_calibration
# Tests for the CalibrationCurve visualizer.
"""
Tests for the CalibrationCurve visualizer (values, errors and drawing).
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from yellowbrick.classifier.calibration import CalibrationCurve, calibration_curve
from yellowbrick.exceptions import ModelError, YellowbrickValueError


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after every test."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def binary_data():
    """A small reproducible binary classification split."""
    X, y = make_classification(n_samples=600, n_features=10, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)


def test_score_sets_attributes(binary_data):
    """score() sets the curve arrays and the Brier score, and returns the score."""
    X_tr, X_te, y_tr, y_te = binary_data
    viz = CalibrationCurve(LogisticRegression(max_iter=1000), n_bins=5)
    viz.fit(X_tr, y_tr)
    result = viz.score(X_te, y_te)

    assert len(viz.prob_true_) == len(viz.prob_pred_)
    assert 1 <= len(viz.prob_true_) <= 5
    assert np.all((viz.prob_true_ >= 0) & (viz.prob_true_ <= 1))
    assert np.all((viz.prob_pred_ >= 0) & (viz.prob_pred_ <= 1))
    assert 0 <= viz.brier_score_ <= 1
    assert result == viz.brier_score_ == viz.score_


def test_brier_matches_sklearn(binary_data):
    """The reported Brier score equals sklearn's brier_score_loss."""
    X_tr, X_te, y_tr, y_te = binary_data
    viz = CalibrationCurve(LogisticRegression(max_iter=1000))
    viz.fit(X_tr, y_tr)
    viz.score(X_te, y_te)

    expected = brier_score_loss(y_te, viz.estimator.predict_proba(X_te)[:, 1])
    assert viz.brier_score_ == pytest.approx(expected)


def test_quantile_strategy(binary_data):
    """The quantile binning strategy works and respects n_bins."""
    X_tr, X_te, y_tr, y_te = binary_data
    viz = CalibrationCurve(
        LogisticRegression(max_iter=1000), n_bins=5, strategy="quantile"
    )
    viz.fit(X_tr, y_tr)
    viz.score(X_te, y_te)
    assert 1 <= len(viz.prob_true_) <= 5


def test_invalid_strategy():
    """An unknown strategy raises a YellowbrickValueError."""
    with pytest.raises(YellowbrickValueError):
        CalibrationCurve(LogisticRegression(), strategy="bogus")


def test_invalid_n_bins():
    """n_bins below 1 raises a YellowbrickValueError."""
    with pytest.raises(YellowbrickValueError):
        CalibrationCurve(LogisticRegression(), n_bins=0)


def test_multiclass_raises():
    """Multiclass targets are rejected, since only binary is supported."""
    X, y = make_classification(
        n_samples=300, n_informative=4, n_classes=3, random_state=0
    )
    viz = CalibrationCurve(LogisticRegression(max_iter=1000))
    viz.fit(X, y)
    with pytest.raises(YellowbrickValueError):
        viz.score(X, y)


def test_requires_predict_proba(binary_data):
    """An estimator without predict_proba raises a ModelError."""
    X_tr, X_te, y_tr, y_te = binary_data
    viz = CalibrationCurve(LinearSVC(max_iter=5000))
    viz.fit(X_tr, y_tr)
    with pytest.raises(ModelError):
        viz.score(X_te, y_te)


def test_draws_reference_and_curve(binary_data):
    """The plot has the diagonal reference line and the model's curve."""
    X_tr, X_te, y_tr, y_te = binary_data
    viz = CalibrationCurve(LogisticRegression(max_iter=1000))
    viz.fit(X_tr, y_tr)
    viz.score(X_te, y_te)

    lines = viz.ax.get_lines()
    labels = [line.get_label() for line in lines]
    assert len(lines) == 2
    assert "Perfectly calibrated" in labels


def test_finalize_sets_labels(binary_data):
    """finalize() adds the title, axis labels and axis limits."""
    X_tr, X_te, y_tr, y_te = binary_data
    viz = CalibrationCurve(LogisticRegression(max_iter=1000))
    viz.fit(X_tr, y_tr)
    viz.score(X_te, y_te)
    viz.finalize()

    assert viz.ax.get_title() == "Calibration Curve for LogisticRegression"
    assert viz.ax.get_xlabel() == "Mean predicted probability"
    assert viz.ax.get_ylabel() == "Fraction of positives"
    assert viz.ax.get_xlim() == (0, 1)


def test_custom_axes_and_string_labels(binary_data):
    """A user-supplied axes is used, and string class labels work."""
    X_tr, X_te, y_tr, y_te = binary_data
    y_tr_str = np.where(y_tr == 1, "yes", "no")
    y_te_str = np.where(y_te == 1, "yes", "no")

    fig, ax = plt.subplots()
    viz = CalibrationCurve(LogisticRegression(max_iter=1000), ax=ax)
    viz.fit(X_tr, y_tr_str)
    str_score = viz.score(X_te, y_te_str)
    assert viz.ax is ax

    int_viz = CalibrationCurve(LogisticRegression(max_iter=1000))
    int_viz.fit(X_tr, y_tr)
    assert str_score == pytest.approx(int_viz.score(X_te, y_te))


def test_quick_method(binary_data):
    """The quick method returns a fitted and finalized visualizer."""
    X_tr, X_te, y_tr, y_te = binary_data
    viz = calibration_curve(
        LogisticRegression(max_iter=1000), X_tr, y_tr, X_te, y_te, show=False
    )
    assert isinstance(viz, CalibrationCurve)
    assert hasattr(viz, "brier_score_")
    assert viz.ax.get_title() == "Calibration Curve for LogisticRegression"


def test_quick_method_without_test_set(binary_data):
    """The quick method scores on the training data if no test set is given."""
    X_tr, X_te, y_tr, y_te = binary_data
    viz = calibration_curve(LogisticRegression(max_iter=1000), X_tr, y_tr, show=False)
    assert 0 <= viz.brier_score_ <= 1
