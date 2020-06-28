.. -*- mode: rst -*-

Classification Report
=====================

The classification report visualizer displays the precision, recall, F1, and support scores for the model. In order to support easier interpretation and problem detection, the report integrates numerical scores with a color-coded heatmap. All heatmaps are in the range ``(0.0, 1.0)`` to facilitate easy comparison of classification models across different classification reports.

=================   =================
Visualizer           :class:`~yellowbrick.classifier.classification_report.ClassificationReport`
Quick Method         :func:`~yellowbrick.classifier.classification_report.classification_report`
Models               Classification
Workflow             Model evaluation
=================   =================

.. plot::
    :context: close-figs
    :alt:  Classification Report

    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.naive_bayes import GaussianNB

    from yellowbrick.classifier import ClassificationReport
    from yellowbrick.datasets import load_occupancy

    # Load the classification dataset
    X, y = load_occupancy()

    # Specify the target classes
    classes = ["unoccupied", "occupied"]

    # Create the training and test data
    tscv = TimeSeriesSplit()
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Instantiate the classification model and visualizer
    model = GaussianNB()
    visualizer = ClassificationReport(model, classes=classes, support=True)

    visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()                       # Finalize and show the figure


The classification report shows a representation of the main classification metrics on a per-class basis. This gives a deeper intuition of the classifier behavior over global accuracy which can mask functional weaknesses in one class of a multiclass problem. Visual classification reports are used to compare classification models to select models that are "redder", e.g. have stronger classification metrics or that are more balanced.

The metrics are defined in terms of true and false positives, and true and false negatives. Positive and negative in this case are generic names for the classes of a binary classification problem. In the example above, we would consider true and false occupied and true and false unoccupied. Therefore a true positive is when the actual class is positive as is the estimated class. A false positive is when the actual class is negative but the estimated class is positive. Using this terminology the metrics are defined as follows:

**precision**
    Precision can be seen as a measure of a classifier's exactness. For each class, it is defined as the ratio of true positives to the sum of true and false positives. Said another way, "for all instances classified positive, what percent was correct?"

**recall**
    Recall is a measure of the classifier's completeness; the ability of a classifier to correctly find all positive instances. For each class, it is defined as the ratio of true positives to the sum of true positives and false negatives. Said another way, "for all instances that were actually positive, what percent was classified correctly?"

**f1 score**
    The F\ :sub:`1` score is a weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. Generally speaking, F\ :sub:`1` scores are lower than accuracy measures as they embed precision and recall into their computation. As a rule of thumb, the weighted average of F\ :sub:`1` should be used to compare classifier models, not global accuracy.


**support**
    Support is the number of actual occurrences of the class in the specified dataset. Imbalanced support in the training data may indicate structural weaknesses in the reported scores of the classifier and could indicate the need for stratified sampling or rebalancing. Support doesn't change between models but instead diagnoses the evaluation process.

.. note:: This example uses ``TimeSeriesSplit`` to split the data into the training and test sets. For more information on this cross-validation method, please refer to the scikit-learn `documentation <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>`_.

Quick Method
------------
The same functionality above can be achieved with the associated quick method ``classification_report``. This method
will build the ``ClassificationReport`` object with the associated arguments, fit it, then (optionally) immediately
show it.

.. plot::
    :context: close-figs
    :alt: classification_report on the occupancy dataset

    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.naive_bayes import GaussianNB

    from yellowbrick.datasets import load_occupancy
    from yellowbrick.classifier import classification_report

    # Load the classification data set
    X, y = load_occupancy()

    # Specify the target classes
    classes = ["unoccupied", "occupied"]

    # Create the training and test data
    tscv = TimeSeriesSplit()
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Instantiate the visualizer
    visualizer = classification_report(
        GaussianNB(), X_train, y_train, X_test, y_test, classes=classes, support=True
    )


API Reference
-------------

.. automodule:: yellowbrick.classifier.classification_report
    :members: ClassificationReport, classification_report
    :undoc-members:
    :show-inheritance:
