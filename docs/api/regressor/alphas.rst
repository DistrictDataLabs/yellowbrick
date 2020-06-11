.. -*- mode: rst -*-

Alpha Selection
===============

Regularization is designed to penalize model complexity, therefore the higher the alpha, the less complex the model, decreasing the error due to variance (overfit). Alphas that are too high on the other hand increase the error due to bias (underfit). It is important, therefore to choose an optimal alpha such that the error is minimized in both directions.

The ``AlphaSelection`` Visualizer demonstrates how different values of alpha influence model selection during the regularization of linear models. Generally speaking, alpha increases the affect of regularization, e.g. if alpha is zero there is no regularization and the higher the alpha, the more the regularization parameter influences the final model.

=================   ==============================
Visualizer           :class:`~yellowbrick.regressor.alphas.AlphaSelection`
Quick Method         :func:`~yellowbrick.regressor.alphas.alphas`
Models               Regression
Workflow             Model selection, Hyperparameter tuning
=================   ==============================

For Estimators *with* Built-in Cross-Validation
-----------------------------------------------

The ``AlphaSelection`` visualizer wraps a "RegressionCV" model and
visualizes the alpha/error curve. Use this visualization to detect if
the model is responding to regularization, e.g. as you increase or
decrease alpha, the model responds and error is decreased. If the
visualization shows a jagged or random plot, then potentially the model
is not sensitive to that type of regularization and another is required
(e.g. L1 or ``Lasso`` regularization).

.. NOTE::
    The ``AlphaSelection`` visualizer requires a "RegressorCV" model, e.g.
    a specialized class that performs cross-validated alpha-selection
    on behalf of the model. See the ``ManualAlphaSelection`` visualizer if
    your regression model does not include cross-validation.

.. plot::
    :context: close-figs
    :alt: Alpha selection on the concrete data set

    import numpy as np

    from sklearn.linear_model import LassoCV
    from yellowbrick.datasets import load_concrete
    from yellowbrick.regressor import AlphaSelection

    # Load the regression dataset
    X, y = load_concrete()

    # Create a list of alphas to cross-validate against
    alphas = np.logspace(-10, 1, 400)

    # Instantiate the linear model and visualizer
    model = LassoCV(alphas=alphas)
    visualizer = AlphaSelection(model)
    visualizer.fit(X, y)
    visualizer.show()

For Estimators *without* Built-in Cross-Validation
--------------------------------------------------

Most scikit-learn ``Estimators`` with ``alpha`` parameters
have a version with built-in cross-validation. However, if the
regressor you wish to use doesn't have an associated "CV" estimator,
or for some reason you would like to specify more control over the
alpha selection process, then you can use the ``ManualAlphaSelection``
visualizer. This visualizer is essentially a wrapper for scikit-learn's
``cross_val_score`` method, fitting a model for each alpha specified.

.. plot::
    :context: close-figs
    :alt: Manual alpha selection on the concrete data set

    import numpy as np

    from sklearn.linear_model import Ridge
    from yellowbrick.datasets import load_concrete
    from yellowbrick.regressor import ManualAlphaSelection

    # Load the regression dataset
    X, y = load_concrete()

    # Create a list of alphas to cross-validate against
    alphas = np.logspace(1, 4, 50)

    # Instantiate the visualizer
    visualizer = ManualAlphaSelection(
        Ridge(),
        alphas=alphas,
        cv=12,
        scoring="neg_mean_squared_error"
    )

    visualizer.fit(X, y)
    visualizer.show()

Quick Methods
-------------

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


The ``ManualAlphaSelection`` visualizer can also be used as a oneliner:

.. plot::
    :context: close-figs
    :alt: manual alphas on the energy dataset

    from sklearn.linear_model import ElasticNet
    from yellowbrick.regressor.alphas import manual_alphas

    from yellowbrick.datasets import load_energy

    # Load dataset
    X, y = load_energy()

    # Instantiate a model
    model = ElasticNet(tol=0.01, max_iter=10000)

    # Use the quick method and immediately show the figure
    manual_alphas(model, X, y, cv=6)


API Reference
-------------

.. automodule:: yellowbrick.regressor.alphas
    :members: AlphaSelection, ManualAlphaSelection, alphas, manual_alphas
    :undoc-members:
    :show-inheritance:
