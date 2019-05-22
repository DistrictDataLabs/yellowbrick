.. -*- mode: rst -*-

ROCAUC
======

A ``ROCAUC`` (Receiver Operating Characteristic/Area Under the Curve) plot allows the user to visualize the tradeoff between the classifier's sensitivity and specificity.

The Receiver Operating Characteristic (ROC) is a measure of a classifier's predictive quality that compares and visualizes the tradeoff between the model's sensitivity and specificity. When plotted, a ROC curve displays the true positive rate on the Y axis and the false positive rate on the X axis on both a global average and per-class basis. The ideal point is therefore the top-left corner of the plot: false positives are zero and true positives are one.

This leads to another metric, area under the curve (AUC), which is a computation of the relationship between false positives and true positives. The higher the AUC, the better the model generally is. However, it is also important to inspect the "steepness" of the curve, as this describes the maximization of the true positive rate while minimizing the false positive rate.

.. plot::
    :context: close-figs
    :alt: ROCAUC Binary Classification

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from yellowbrick.classifier import ROCAUC
    from yellowbrick.datasets import load_spam

    # Load the classification dataset
    X, y = load_spam()

    # Create the training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Instantiate the visualizer with the classification model
    model = LogisticRegression(multi_class="auto", solver="liblinear")
    visualizer = ROCAUC(model, classes=["not_spam", "is_spam"])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.poof()                       # Draw/show/poof the data


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

    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

    from yellowbrick.classifier import ROCAUC
    from yellowbrick.datasets import load_game

    # Load multi-class classification dataset
    X, y = load_game()

    # Encode the non-numeric columns
    X = OrdinalEncoder().fit_transform(X)
    y = LabelEncoder().fit_transform(y)

    # Create the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Instaniate the classification model and visualizer
    model = RidgeClassifier()
    visualizer = ROCAUC(model, classes=["win", "loss", "draw"])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.poof()                       # Draw/show/poof the data

.. warning::
    The target ``y`` must be numeric for this figure to work, or update to the latest version of sklearn.

By default with multi-class ROCAUC visualizations, a curve for each class is plotted, in addition to the micro- and macro-average curves for each class. This enables the user to inspect the tradeoff between sensitivity and specificity on a per-class basis. Note that for multi-class ``ROCAUC``, at least one of the ``micro``, ``macro``, or ``per_class`` parameters must be set to ``True`` (by default, all are set to ``True``).


API Reference
-------------

.. automodule:: yellowbrick.classifier.rocauc
    :members: ROCAUC
    :undoc-members:
    :show-inheritance:
