.. -*- mode: rst -*-

Classification Report
=====================

The classification report visualizer displays the precision, recall, and F1 scores for the model. In order to support easier interpretation and problem detection, the report integrates numerical scores with a color-coded heatmap. All heatmaps are in the range ``(0.0, 1.0)`` to facilitate easy comparison of classification models across different classification reports.

.. code:: python

    from sklearn.model_selection import train_test_split

    # Load the classification data set
    data = load_data("occupancy")

    # Specify the features of interest and the classes of the target
    features = [
        "temperature", "relative humidity", "light", "C02", "humidity"
    ]
    classes = ["unoccupied", "occupied"]

    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.occupancy.as_matrix()

    # Create the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

.. code:: python

    from sklearn.naive_bayes import GaussianNB
    from yellowbrick.classifier import ClassificationReport

    # Instantiate the classification model and visualizer
    bayes = GaussianNB()
    visualizer = ClassificationReport(bayes, classes=classes)

    visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    g = visualizer.poof()             # Draw/show/poof the data



.. image:: images/classification_report.png


The classification report shows a representation of the main classification metrics on a per-class basis. This gives a deeper intuition of the classifier behavior over global accuracy which can mask functional weaknesses in one class of a multiclass problem. Visual classification reports are used to compare classification models to select models that are "redder", e.g. have stronger classification metrics or that are more balanced.

The metrics are defined in terms of true and false positives, and true and false negatives. Positive and negative in this case are generic names for the classes of a binary classification problem. In the example above, we would consider true and false occupied and true and false unoccupied. Therefore a true positive is when the actual class is positive as is the estimated class. A false positive is when the actual class is negative but the estimated class is positive. Using this terminology the meterics are defined as follows:

**precision**
    Precision is the ability of a classiifer not to label an instance positive that is actually negative. For each class it is defined as as the ratio of true positives to the sum of true and false positives. Said another way, "for all instances classified positive, what percent was correct?"

**recall**
    Recall is the ability of a classifier to find all positive instances. For each class it is defined as the ratio of true positives to the sum of true positives and false negatives. Said another way, "for all instances that were actually positive, what percent was classified correctly?"

**f1 score**
    The F\ :sub:`1` score is a weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. Generally speaking, F\ :sub:`1` scores are lower than accuracy measures as they embed precision and recall into their computation. As a rule of thumb, the weighted average of F\ :sub:`1` should be used to compare classifier models, not global accuracy.

.. caution::
    Support is omitted from the classification metrics because it is difficult to scale this on a heatmap. In a future release we will add a support feature that describes the support as the percentage of the total number of instances.


API Reference
-------------

.. automodule:: yellowbrick.classifier.classification_report
    :members: ClassificationReport
    :undoc-members:
    :show-inheritance:
