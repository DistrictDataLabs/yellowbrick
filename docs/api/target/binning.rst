.. -*- mode: rst -*-

Balanced Binning Reference
==========================

Frequently, machine learning problems in the real world suffer from the curse of dimensionality; you have fewer training instances than you'd like and the predictive signal is distributed (often unpredictably!) across many different features.

Sometimes when the your target variable is continuously-valued, there simply aren't enough instances to predict these values to the precision of regression. In this case, we can sometimes transform the regression problem into a classification problem by binning the continuous values into makeshift classes.

To help the user select the optimal number of bins, the ``BalancedBinningReference`` visualizer takes the target variable ``y`` as input and generates a histogram with vertical lines indicating the recommended value points to ensure that the data is evenly distributed into each bin.


.. code:: python

    from yellowbrick.target import BalancedBinningReference

    # Load the a regression data set
    data = load_data("concrete")

    # Extract the target of interest
    y = data["strength"]

    # Instantiate the visualizer
    visualizer = BalancedBinningReference()

    visualizer.fit(y)          # Fit the data to the visualizer
    visualizer.poof()          # Draw/show/poof the data


.. image:: images/balanced_binning_reference.png

.. seealso::

    To learn more, please read Rebecca Bilbro's article `"Creating Categorical Variables from Continuous Data." <https://rebeccabilbro.github.io/better-binning>`_


API Reference
-------------

.. automodule:: yellowbrick.target.binning
    :members: BalancedBinningReference
    :undoc-members:
    :show-inheritance:
