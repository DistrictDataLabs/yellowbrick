.. -*- mode: rst -*-

statsmodels Visualizers
=======================

`statsmodels <https://www.statsmodels.org/stable/index.html>`_ is a Python library that provides utilities for the estimation of several statistical models and includes extensive results and metrics for each estimator. In particular, statsmodels excels at generalized linear models (GLMs) which are far superior to scikit-learn's implementation of ordinary least squares.

This contrib module allows statsmodels users to take advantage of Yellowbrick visualizers by creating a wrapper class that implements the scikit-learn ``BaseEstimator``. Using the wrapper class, statsmodels can be passed directly to many visualizers, customized for the scoring and metric functionality required.

.. warning:: The statsmodel wrapper is currently a prototype and as such is currently a bit trivial. Many options and extra functionality such as weights are not currently handled. We are actively looking for statsmodels users to contribute to this package!

Using the statsmodels wrapper:

.. code:: python

    import statsmodels.api as sm

    from functools import partial
    from yellowbrick.regressor import ResidualsPlot
    from yellowbrick.contrib.statsmodels import StatsModelsWrapper

    glm_gaussian_partial = partial(sm.GLM, family=sm.families.Gaussian())
    model = StatsModelsWrapper(glm_gaussian_partial)

    viz = ResidualsPlot(model)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show()

You can also use fitted estimators with the wrapper to avoid having to pass a partial function:

.. code:: python

    from yellowbrick.regressor import prediction_error

    # Create the OLS model
    model = sm.OLS(y, X)

    # Get the detailed results
    results = model.fit()
    print(results.summary())

    # Visualize the prediction error
    prediction_error(StatsModelWrapper(model), X, y, is_fitted=True)

This example also shows the use of a Yellowbrick oneliner, which is often more suited to the analytical style of statsmodels.


API Reference
-------------

.. automodule:: yellowbrick.contrib.statsmodels.base
    :members: StatsModelsWrapper
    :undoc-members:
    :show-inheritance:
