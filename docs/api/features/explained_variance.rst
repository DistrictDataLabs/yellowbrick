.. -*- mode: rst -*-

Explained Variance
==================

=================   =================
Visualizer           :class:`~yellowbrick.features.explained_variance.ExplainedVariance`
Quick Method         :func:`~yellowbrick.features.explained_variance.explained_variance`
Models               Decomposition
Workflow             Feature Engineering
=================   =================

.. plot::
    :context: close-figs
    :alt: Explained variance quick method on the credit dataset

    from yellowbrick.datasets import load_credit
    from yellowbrick.features import ExplainedVariance

    # Specify the features of interest and the target
    X, _ = load_credit()

    # Instantiate the visualizer, fit and transform the data
    oz = ExplainedVariance()
    oz.fit_transform(X)
    oz.show()


Quick Method
------------

The same functionality above can be achieved with the associated quick method ``explained_variance``. This method will build the ``ExplainedVariance`` visualizer with the associated arguments, fit it, then (optionally) immediately show it.

.. plot::
    :context: close-figs
    :alt: Explained variance quick method on the concrete dataset

    from yellowbrick.datasets import load_concrete
    from yellowbrick.features import explained_variance

    # Specify the features of interest and the target
    X, _ = load_concrete()

    # Determine the optimal number of components
    oz = explained_variance(X)


API Reference
-------------

.. automodule:: yellowbrick.features.explained_variance
    :members: ExplainedVariance, explained_variance
    :undoc-members:
    :show-inheritance:
