.. -*- mode: rst -*-

PrePredict Estimators
=====================

Occassionally it is useful to be able to use predictions made during an inferencing workflow that does not involve Yellowbrick, for example when the inferencing process requires extra compute resources such as a cluster or when the model takes a very long time to train and inference. In other instances there are models that Yellowbrick simply does not support, even with the :doc:`third-party estimator wrapper <wrapper>` or the results may have been collected from some source out of your control.

Some Yellowbrick visualizers are still able to create visual diagnostics with predictions already made using the contrib library ``PrePredict`` estimator, which is a simple wrapper around some data and an estimator type. Although not quite as straight forward as a scikit-learn metric in the form ``metric(y_true, y_pred)``, this estimator allows Yellowbrick to be used in the cases described above, an example is below:

.. code:: python

    # Import the prepredict estimator and a Yellowbrick visualizer
    from yellowbrick.contrib.prepredict import PrePredict, CLASSIFIER
    from yellowbrick.classifier import classification_report

    # Instantiate the estimator with the pre-predicted data
    model = PrePredict(y_pred, CLASSIFIER)

    # Use the visualizer, setting X to None since it is not required
    oz = classification_report(model, None, y_test)
    oz.show()

.. warning:: Many Yellowbrick visualizers inspect the estimator for learned attributes in order to deliver rich diagnostics. You may run into visualizers that cannot use the prepredict method, or you can manually set attributes on the ``PrePredict`` estimator with the learned attributes the visualizer requires.

In the case where you've saved pre-predicted data from disk, the ``PrePredict`` estimator can load it using ``np.load``. A full workflow is described below:

.. code:: python

    # Phase one: fit your estimator, make inferences, and save the inferences to disk
    np.save("y_pred.npy", y_pred)

    # Import the prepredict estimator and a Yellowbrick visualizer
    from yellowbrick.contrib.prepredict import PrePredict, REGRESSOR
    from yellowbrick.regressor import prediction_error

    # Instantiate the estimator with the pre-predicted data and pass a path to where
    # the data has been saved on disk.
    model = PrePredict("y_pred.npy", REGRESSOR)

    # Use the visualizer, setting X to None since it is not required
    oz = prediction_error(model, X_test, y_test)
    oz.show()

The ``PrePredict`` estimator can use a callable function to return pre-predicted data, a ``str``, file-like object, or ``pathlib.Path`` to load from disk using ``np.load``, otherwise it simply returns the data it wraps. See the API reference for more details.

API Reference
-------------

.. automodule:: yellowbrick.contrib.prepredict
    :members: PrePredict
    :undoc-members:
    :show-inheritance: