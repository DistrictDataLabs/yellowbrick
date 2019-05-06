.. -*- mode: rst -*-

ROCAUC
======

A ``ROCAUC`` (Receiver Operating Characteristic/Area Under the Curve) plot allows the user to visualize the tradeoff between the classifier's sensitivity and specificity.

The Receiver Operating Characteristic (ROC) is a measure of a classifier's predictive quality that compares and visualizes the tradeoff between the model's sensitivity and specificity. When plotted, a ROC curve displays the true positive rate on the Y axis and the false positive rate on the X axis on both a global average and per-class basis. The ideal point is therefore the top-left corner of the plot: false positives are zero and true positives are one.

This leads to another metric, area under the curve (AUC), which is a computation of the relationship between false positives and true positives. The higher the AUC, the better the model generally is. However, it is also important to inspect the "steepness" of the curve, as this describes the maximization of the true positive rate while minimizing the false positive rate.

.. plot::
    :context: close-figs
    :alt: ROCAUC Binary Classification

    from yellowbrick.classifier import ROCAUC
    from sklearn.linear_model import LogisticRegression
    from yellowbrick.datasets import load_occupancy

    # Load the classification data set
    X, y = load_occupancy()

    # Specify the classes of the target
    classes = ["unoccupied", "occupied"]

    # Transform Pandas to numpy
    X = X.to_numpy()
    y = y.to_numpy()

    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        break # taking only the first split as the example data to visualize.

    # Instantiate the visualizer with the classification model
    visualizer = ROCAUC(LogisticRegression(
        multi_class="auto", solver="liblinear"
        ), classes=classes
    )
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.poof()             # Draw/show/poof the data


.. warning::
    Versions of Yellowbrick =< v0.8 had a `bug <https://github.com/DistrictDataLabs/yellowbrick/blob/develop/examples/rebeccabilbro/rocauc_bug_research.ipynb>`_ 
    that triggered an ``IndexError`` when attempting binary classification using 
    a Scikit-learn-style estimator with only a ``decision_function``. This has been
    fixed as of v0.9, where the ``micro``, ``macro``, and ``per-class`` parameters of 
    ``ROCAUC`` are set to ``False`` for such classifiers.


Multi-class ROCAUC Curves
-------------------------

Yellowbrick's ``ROCAUC`` Visualizer does allow for plotting multiclass classification curves.
ROC curves are typically used in binary classification, and in fact the Scikit-Learn ``roc_curve`` metric is only able to perform metrics for binary classifiers. Yellowbrick addresses this by binarizing the output (per-class) or to use one-vs-rest (micro score) or one-vs-all (macro score) strategies of classification.

.. plot::
    :context: close-figs
    :alt: ROCAUC multiclass classification curves

    from sklearn.model_selection import train_test_split
    from yellowbrick.classifier import ROCAUC
    from yellowbrick.datasets import load_game
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

    # Load multi-class classification dataset
    X, y = load_game()

    classes = ["win", "loss", "draw"]

    # Encode the non-numeric columns
    X = OrdinalEncoder().fit_transform(X)

    y = LabelEncoder().fit_transform(y)

    # Create the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    visualizer = ROCAUC(RidgeClassifier(), classes=classes)

    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.poof()             # Draw/show/poof the data

.. warning::
    The target ``y`` must be numeric for this figure to work, or update to the latest version of sklearn.

By default with multi-class ROCAUC visualizations, a curve for each class is plotted, in addition to the micro- and macro-average curves for each class. This enables the user to inspect the tradeoff between sensitivity and specificity on a per-class basis. Note that for multi-class ``ROCAUC``, at least one of the ``micro``, ``macro``, or ``per_class`` parameters must be set to ``True`` (by default, all are set to ``True``).


API Reference
-------------

.. automodule:: yellowbrick.classifier.rocauc
    :members: ROCAUC
    :undoc-members:
    :show-inheritance:
