.. -*- mode: rst -*-

Regression Visualizers
======================

Regression models attempt to predict a target in a continuous space.
Regressor score visualizers display the instances in model space to
better understand how the model is making predictions. We currently have
implemented three regressor evaluations:

-  :doc:`residuals`: plot the difference between the expected and actual
   values
-  :doc:`peplot`: plot the expected vs. actual values in model space
-  :doc:`alphas`: visual tuning of regularization hyperparameters

Estimator score visualizers *wrap* Scikit-Learn estimators and expose
the Estimator API such that they have ``fit()``, ``predict()``, and
``score()`` methods that call the appropriate estimator methods under
the hood. Score visualizers can wrap an estimator and be passed in as
the final step in a ``Pipeline`` or ``VisualPipeline``.

.. code:: python

    # Regression Evaluation Imports

    from sklearn.linear_model import Ridge, Lasso
    from sklearn.model_selection import train_test_split

    from yellowbrick.regressor import PredictionError, ResidualsPlot
    from yellowbrick.regressor.alphas import AlphaSelection


.. toctree::
   :maxdepth: 2

   residuals
   peplot
   alphas
