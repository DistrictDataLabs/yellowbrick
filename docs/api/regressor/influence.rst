.. -*- mode: rst -*-

Cook's Distance
===============

Cook's Distance is a measure of an observation or instances' influence on a linear
regression. Instances with a large influence may be outliers and datasets that have a
large number of highly influential points might not be good predictors to fit linear
models. The ``CooksDistance`` visualizer shows a stem plot of all instances by index
and their associated distance score, along with a heuristic threshold to quickly show
what percent of the dataset may be impacting OLS regression models.

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
    visualizer.poof()


API Reference
-------------

.. automodule:: yellowbrick.regressor.influence
    :members: CooksDistance
    :undoc-members:
    :show-inheritance:

