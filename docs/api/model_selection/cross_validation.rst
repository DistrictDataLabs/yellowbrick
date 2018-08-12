.. -*- mode: rst -*-

Cross Validation Scores
=======================

Generally we determine whether a given model is optimal by looking at it's F1, precision, recall, and accuracy (for classification), or it's coefficient of determination (R2) and error (for regression). However, real world data is often distributed somewhat unevenly, meaning that the fitted model is likely to perform better on some sections of the data than on others. Yellowbrick's ``CVScores`` visualizer enables us to visually explore these variations in performance using different cross validation strategies.

Cross Validation
################

Cross-validation starts by shuffling the data (to prevent any unintentional ordering errors) and splitting it into `k` folds. Then `k` models are fit on :math:`\frac{k-1} {k}` of the data (called the training split) and evaluated on :math:`\frac {1} {k}` of the data (called the test split). The results from each evaluation are averaged together for a final score, then the final model is fit on the entire dataset for operationalization.

.. image:: images/cross_validation.png

In Yellowbrick, the ``CVScores`` visualizer displays cross-validated scores as a bar chart (one bar for each fold) with the average score across all folds plotted as a horizontal dotted line.

Classification
--------------

In the following example we show how to visualize cross-validated scores for a classification model. After loading a ``DataFrame``, we create a ``StratifiedKFold`` cross-validation strategy to ensure all of our classes in each split are represented with the same proportion. We then fit the ``CVScores`` visualizer using the ``f1_weighted`` scoring metric as opposed to the default metric, accuracy, to get a better sense of the relationship of precision and recall in our classifier across all of our folds.

.. code:: python

    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import StratifiedKFold

    from yellowbrick.model_selection import CVScores


    room = load_data("occupancy")

    features = ["temperature", "relative humidity", "light", "C02", "humidity"]

    # Extract the numpy arrays from the data frame
    X = room[features].values
    y = room.occupancy.values

    # Create a new figure and axes
    _, ax = plt.subplots()

    # Create a cross-validation strategy
    cv = StratifiedKFold(12)

    # Create the cv score visualizer
    oz = CVScores(
        MultinomialNB(), ax=ax, cv=cv, scoring='f1_weighted'
    )

    oz.fit(X, y)
    oz.poof()


Our resulting visualization shows that while our average cross-validation score is quite high, there are some splits for which our fitted ``MultinomialNB`` classifier performs significantly less well.


.. image:: images/cv_scores_classifier.png


Regression
----------

In this next example we show how to visualize cross-validated scores for a regression model. After loading our energy data into a ``DataFrame``, we instantiate a simple ``KFold`` cross-validation strategy. We then fit the ``CVScores`` visualizer using the ``r2`` scoring metric, to get a sense of the coefficient of determination for our regressor across all of our folds.

.. code:: python

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold


    energy = load_data("energy")

    targets = ["heating load", "cooling load"]
    features = [col for col in energy.columns if col not in targets]

    X = energy[features]
    y = energy[targets[1]]

    # Create a new figure and axes
    _, ax = plt.subplots()

    cv = KFold(12)

    oz = CVScores(
        Ridge(), ax=ax, cv=cv, scoring='r2'
    )

    oz.fit(X, y)
    oz.poof()


As with our classification ``CVScores`` visualization, our regression visualization suggests that our ``Ridge`` regressor performs very well (e.g. produces a high coefficient of determination) across nearly every fold, resulting in another fairly high overall R2 score.

.. image:: images/cv_scores_regressor.png


API Reference
-------------

.. automodule:: yellowbrick.model_selection.cross_validation
    :members: CVScores
    :undoc-members:
    :show-inheritance:
