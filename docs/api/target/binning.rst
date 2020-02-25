.. -*- mode: rst -*-

Balanced Binning Reference
==========================

Frequently, machine learning problems in the real world suffer from the curse of dimensionality; you have fewer training instances than you'd like and the predictive signal is distributed (often unpredictably!) across many different features.

Sometimes when the your target variable is continuously-valued, there simply aren't enough instances to predict these values to the precision of regression. In this case, we can sometimes transform the regression problem into a classification problem by binning the continuous values into makeshift classes.

To help the user select the optimal number of bins, the ``BalancedBinningReference`` visualizer takes the target variable ``y`` as input and generates a histogram with vertical lines indicating the recommended value points to ensure that the data is evenly distributed into each bin.

=================   ==============================
Visualizer           :class:`~yellowbrick.target.binning.BalancedBinningReference`
Quick Method         :func:`~yellowbrick.target.binning.balanced_binning_reference`
Models               Classification
Workflow             Feature analysis, Target analysis, Model selection
=================   ==============================

.. plot::
    :context: close-figs
    :alt: BalancedBinningReference on concrete dataset

    from yellowbrick.datasets import load_concrete
    from yellowbrick.target import BalancedBinningReference

    # Load the concrete dataset
    X, y = load_concrete()

    # Instantiate the visualizer
    visualizer = BalancedBinningReference()

    visualizer.fit(y)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure


Quick Method
------------

The same functionality above can be achieved with the associated quick method ``balanced_binning_reference``. This method will build the ``BalancedBinningReference`` object with the associated arguments, fit it, then (optionally) immediately show it.

.. plot::
    :context: close-figs
    :alt: balanced_binning_reference on the concrete dataset

    from yellowbrick.datasets import load_concrete
    from yellowbrick.target import balanced_binning_reference

    # Load the dataset
    X, y = load_concrete()

    # Use the quick method and immediately show the figure
    balanced_binning_reference(y)


.. seealso::

    To learn more, please read Rebecca Bilbro's article `"Creating Categorical Variables from Continuous Data." <https://rebeccabilbro.github.io/better-binning>`_


API Reference
-------------

.. automodule:: yellowbrick.target.binning
    :members: BalancedBinningReference, balanced_binning_reference
    :undoc-members:
    :show-inheritance:
