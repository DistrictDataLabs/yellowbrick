.. -*- mode: rst -*-

Calibration Curve
=================

A calibration curve (also called a reliability diagram) shows whether a binary classifier's predicted probabilities match what actually happens. Predictions are grouped into bins, and for each bin the mean predicted probability is plotted against the observed fraction of positives. A perfectly calibrated model lies on the diagonal. The Brier score, shown in the legend, summarizes the overall error (lower is better).

=================   ===============================================================
Visualizer           :class:`~yellowbrick.classifier.calibration.CalibrationCurve`
Quick Method         :func:`~yellowbrick.classifier.calibration.calibration_curve`
Models               Classification (binary, needs ``predict_proba``)
Workflow             Model evaluation
=================   ===============================================================

.. plot::
    :context: close-figs
    :alt: Calibration curve for a LogisticRegression model

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from yellowbrick.classifier import CalibrationCurve

    X, y = make_classification(n_samples=2000, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    viz = CalibrationCurve(LogisticRegression(max_iter=1000))
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show()

Binning strategy
----------------

By default the probability range is split into ``n_bins`` equal-width bins (``strategy="uniform"``). With ``strategy="quantile"`` each bin holds the same number of samples, which is useful when most predictions fall in a narrow range.

.. code:: python

    viz = CalibrationCurve(LogisticRegression(max_iter=1000), n_bins=5, strategy="quantile")

Limitations
-----------

Only binary classification is supported, and the estimator must implement ``predict_proba``. The positive class is the second entry of ``estimator.classes_``.

API Reference
-------------

.. automodule:: yellowbrick.classifier.calibration
    :members: CalibrationCurve, calibration_curve
    :undoc-members:
    :show-inheritance:
