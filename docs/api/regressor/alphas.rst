.. -*- mode: rst -*-

Alpha Selection
===============

Regularization is designed to penalize model complexity, therefore the higher the alpha, the less complex the model, decreasing the error due to variance (overfit). Alphas that are too high on the other hand increase the error due to bias (underfit). It is important, therefore to choose an optimal alpha such that the error is minimized in both directions.

The AlphaSelection Visualizer demonstrates how different values of alpha influence model selection during the regularization of linear models. Generally speaking, alpha increases the affect of regularization, e.g. if alpha is zero there is no regularization and the higher the alpha, the more the regularization parameter influences the final model.

=================   ==============================
Visualizer           :class:`~yellowbrick.regressor.alphas.AlphaSelection`
Quick Method         :func:`~yellowbrick.regressor.alphas.alphas`
Models               Regression
Workflow             Model selection, Hyperparameter tuning
=================   ==============================


.. plot::
    :context: close-figs
    :alt: Alpha selection on the concrete data set

    import numpy as np

    from sklearn.linear_model import LassoCV
    from yellowbrick.regressor import AlphaSelection
    from yellowbrick.datasets import load_concrete

    # Load the regression dataset
    X, y = load_concrete()

    # Create a list of alphas to cross-validate against
    alphas = np.logspace(-10, 1, 400)

    # Instantiate the linear model and visualizer
    model = LassoCV(alphas=alphas)
    visualizer = AlphaSelection(model)
    visualizer.fit(X, y)
    visualizer.show()


Quick Method
------------

The same functionality above can be achieved with the associated quick method `alphas`. This method will build the ``AlphaSelection`` Visualizer object with the associated arguments, fit it, then (optionally) immediately show it.


.. plot::
    :context: close-figs
    :alt: alphas on the energy dataset

    from sklearn.linear_model import LassoCV
    from yellowbrick.regressor.alphas import alphas

    from yellowbrick.datasets import load_energy

    # Load dataset
    X, y = load_energy()

    # Use the quick method and immediately show the figure
    alphas(LassoCV(random_state=0), X, y)


API Reference
-------------

.. automodule:: yellowbrick.regressor.alphas
    :members: AlphaSelection, ManualAlphaSelection, alphas
    :undoc-members:
    :show-inheritance:
