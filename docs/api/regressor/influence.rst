.. -*- mode: rst -*-

Cook's Distance
===============

Cook's Distance is a measure of an observation or instances' influence on a linear
regression. Instances with a large influence may be outliers and datasets that have a
large number of highly influential points might not be good predictors to fit linear
models. The ``CooksDistance`` visualizer shows a stem plot of all instances by index
and their associated distance score, along with a heuristic threshold to quickly show
what percent of the dataset may be impacting OLS regression models.

=================   ==============================
Visualizer           :class:`~yellowbrick.regressor.influence.CooksDistance`
Quick Method         :func:`~yellowbrick.regressor.influence.cooks_distance`
Models               General Linear Models
Workflow             Dataset/Sensitivity Analysis
=================   ==============================

.. plot::
    :context: close-figs
    :alt: Cook's distance using concrete dataset

    from yellowbrick.regressor import CooksDistance
    from yellowbrick.datasets import load_concrete

    # Load the regression dataset
    X, y = load_concrete()

    # Instantiate and fit the visualizer
    visualizer = CooksDistance()
    visualizer.fit(X, y)
    visualizer.show()

Quick Method
------------

Similar functionality as above can be achieved in one line using the associated quick method, ``class_prediction_error``. This method will instantiate and fit a ``ClassPredictionError`` visualizer on the training data, then will score it on the optionally provided test data (or the training data if it is not provided).

.. plot::
    :context: close-figs
    :alt: cooks_distance quick method

    from yellowbrick.datasets import load_concrete
    from yellowbrick.regressor import cooks_distance

    # Load the regression dataset
    X, y = load_concrete()

    # Instantiate and fit the visualizer
    cooks_distance(
        X, y,
        draw_threshold=True,
        linefmt="C0-", markerfmt=","
    )


API Reference
-------------

.. automodule:: yellowbrick.regressor.influence
    :members: CooksDistance, cooks_distance
    :undoc-members:
    :show-inheritance:

