.. -*- mode: rst -*-

Feature Dropping Curve
=============================

 =================   =====================
 Visualizer           :class:`~yellowbrick.model_selection.dropping_curve.DroppingCurve`
 Quick Method         :func:`~yellowbrick.model_selection.dropping_curve.dropping_curve`
 Models               Classification, Regression, Clustering
 Workflow             Model Selection
 =================   =====================

A feature dropping curve (FDC) shows the relationship between the score and the number of features used.
This visualizer randomly drops input features, showing how the estimator benefits from additional features of the same type.
For example, how many air quality sensors are needed across a city to accurately predict city-wide pollution levels?

Feature dropping curves helpfully complement :doc:`rfecv` (RFECV).
In the air quality sensor example, RFECV finds which sensors to keep in the specific city.
Feature dropping curves estimate how many sensors a similar-sized city might need to track pollution levels.

Feature dropping curves are common in the field of neural decoding, where they are called `neuron dropping curves <https://dx.doi.org/10.3389%2Ffnsys.2014.00102>`_ (`example <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8293867/figure/F3/>`_, panels C and H).
Neural decoding research often quantifies how performance scales with neuron (or electrode) count.
Because neurons do not correspond directly between participants, we use random neuron subsets to simulate what performance to expect when recording from other participants.

To show how this works in practice, consider an image classification example using `handwritten digits <https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits>`_.

.. plot::
    :context: close-figs
    :alt: Dropping Curve on the digits dataset

    from sklearn.svm import SVC
    from sklearn.datasets import load_digits

    from yellowbrick.model_selection import DroppingCurve

    # Load dataset
    X, y = load_digits(return_X_y=True)

    # Initialize visualizer with estimator
    visualizer = DroppingCurve(SVC())

    # Fit the data to the visualizer
    visualizer.fit(X, y)
    # Finalize and render the figure
    visualizer.show()

This figure shows an input feature dropping curve.
Since the features are informative, the accuracy increases with more larger feature subsets.
The shaded area represents the variability of cross-validation, one standard deviation above and below the mean accuracy score drawn by the curve.

The visualization can be interpreted as the performance if we knew some image pixels were corrupted.
As an alternative interpretation, the dropping curve roughly estimates the accuracy if the image resolution was downsampled.

Quick Method
------------
The same functionality can be achieved with the associated quick method ``dropping_curve``. This method will build the ``DroppingCurve`` with the associated arguments, fit it, then (optionally) immediately show the visualization.

.. plot::
    :context: close-figs
    :alt: Dropping Curve Quick Method on the digits dataset

    from sklearn.svm import SVC
    from sklearn.datasets import load_digits

    from yellowbrick.model_selection import dropping_curve

    # Load dataset
    X, y = load_digits(return_X_y=True)

    dropping_curve(SVC(), X, y)


API Reference
-------------

.. automodule:: yellowbrick.model_selection.dropping_curve
    :members: DroppingCurve, dropping_curve
    :undoc-members:
    :show-inheritance:
