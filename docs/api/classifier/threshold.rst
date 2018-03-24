.. -*- mode: rst -*-

Discrimination Threshold
========================

.. caution:: This visualizer only works for *binary* classification.

A visualization of precision, recall, f1 score, and "queue rate" with respect to the "discrimination threshold" of a binary classifier. The *discrimination threshold* is the probability or score at which the positive class is chosen over the negative class. Generally this is set to 50% but the threshold can be adjusted to increase or decrease the sensitivity to false positives or to other application factors.

.. code:: python

    # Load a binary classification dataset
    data = load_data("spam")
    target = "is_spam"
    features = [col for col in data.columns if col != target]

    # Extract the instances and target from the dataset
    X = data[features]
    y = data[target]

.. code:: python

    from sklearn.linear_model import LogisticRegression
    from yellowbrick.classifier import DiscriminationThreshold

    # Instantiate the classification model and visualizer
    logistic = LogisticRegression()
    visualizer = DiscriminationThreshold(logistic)

    visualizer.fit(X, y)  # Fit the training data to the visualizer
    visualizer.poof()     # Draw/show/poof the data

.. image:: images/spam_discrimination_threshold.png




API Reference
-------------

.. automodule:: yellowbrick.classifier.threshold
    :members: DiscriminationThreshold
    :undoc-members:
    :show-inheritance:
