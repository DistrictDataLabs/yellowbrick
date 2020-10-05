.. -*- mode: rst -*-

Using Third-Party Estimators
============================

Many machine learning libraries implement the scikit-learn estimator API to easily integrate alternative optimization or decision methods into a data science workflow. Because of this, it seems like it should be simple to drop in a non-scikit-learn estimator into a Yellowbrick visualizer, and in principle, it is. However, the reality is a bit more complicated.

Yellowbrick visualizers often utilize more than just the method interface of estimators (e.g. ``fit()`` and ``predict()``), relying on the learned attributes (object properties with a single underscore suffix, e.g. ``coef_``). The issue is that when a third-party estimator does not expose these attributes, truly gnarly exceptions and tracebacks occur. Yellowbrick is meant to aid machine learning diagnostics reasoning, therefore instead of just allowing drop-in functionality that may cause confusion, we've created a wrapper functionality that is a bit kinder with it's messaging.

But first, an example.

.. code:: python

    # Import the wrap function and a Yellowbrick visualizer
    from yellowbrick.contrib.wrapper import wrap
    from yellowbrick.model_selection import feature_importances

    # Instantiate the third party estimator and wrap it, optionally fitting it
    model = wrap(ThirdPartyEstimator())
    model.fit(X_train, y_train)

    # Use the visualizer
    oz = feature_importances(model, X_test, y_test, is_fitted=True)

The ``wrap`` function initializes the third party model as a ``ContribEstimator``, which passes through all functionality to the underlying estimator, however if an error occurs, the exception that will be raised looks like:

.. code:: text

    yellowbrick.exceptions.YellowbrickAttributeError: estimator is missing the 'fit'
    attribute, which is required for this visualizer - please see the third party
    estimators documentation.

Some estimators are required to pass type checking, for example the estimator must be a classifier, regressor, clusterer, density estimator, or outlier detector. A second argument can be passed to the ``wrap`` function declaring the type of estimator:

.. code:: python

    from yellowbrick.classifier import precision_recall_curve
    from yellowbrick.contrib.wrapper import wrap, CLASSIFIER

    model = wrap(ThirdPartyClassifier(), CLASSIFIER)
    precision_recall_curve(model, X, y)

Or you can simply use the wrap helper functions of the specific type:

.. code:: python

    from yellowbrick.contrib.wrapper import regressor, classifier, clusterer
    from yellowbrick.regressor import prediction_error
    from yellowbrick.classifier import classification_report
    from yellowbrick.cluster import intercluster_distance

    reg = regressor(ThirdPartyRegressor())
    prediction_error(reg, X, y)

    clf = classifier(ThirdPartyClassifier())
    classification_report(clf, X, y)

    ctr = clusterer(ThirdPartyClusterer())
    intercluster_distance(ctr, X)


So what should you do if a required attribute is missing from your estimator? The simplest and quickest thing to do is to subclass ``ContribEstimator`` and add the required functionality.

.. code:: python

    from yellowbrick.contrib.wrapper import ContribEstimator, CLASSIFIER

    class MyWrapper(ContribEstimator):

        _estimator_type = CLASSIFIER

        @property
        def feature_importances_(self):
            return self.estimator.tree_feature_importances()


    model = MyWrapper(ThirdPartyEstimator()
    feature_importances(model, X, y)


This is certainly less than ideal - but we'd welcome a contrib PR to add more native functionality to Yellowbrick!

Tested Libraries
----------------

The following libraries have been tested with the Yellowbrick wrapper.

- `xgboost <https://xgboost.readthedocs.io/en/latest/index.html>`_: both the ``XGBRFRegressor`` and ``XGBRFClassifier`` have been tested with Yellowbrick both with and without the wrapper functionality.
- `CatBoost <https://catboost.ai/>`_: the ``CatBoostClassifier`` has been tested with the ``ClassificationReport`` visualizer.

The following libraries have been partially tested and will likely work without too much additional effort:

- `cuML <https://github.com/rapidsai/cuml>`_: it is likely that clustering, classification, and regression cuML estimators will work with Yellowbrick visualizers. However, the cuDF datasets have not been tested with Yellowbrick.
- `Spark MLlib <https://spark.apache.org/docs/latest/ml-guide.html>`_: The Spark DataFrame API and estimators should work with Yellowbrick visualizers in a local notebook context after collection.

.. note:: If you have used a Python machine learning library not listed here with Yellowbrick, please let us know - we'd love to add it to the list! Also if you're using a library that is not wholly compatible, please open an issue so that we can explore how to integrate it with the ``yellowbrick.contrib`` module!

API Reference
-------------

.. automodule:: yellowbrick.contrib.wrapper
    :members: wrap, classifier, regressor, clusterer, ContribEstimator
    :undoc-members:
    :show-inheritance: