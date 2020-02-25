.. -*- mode: rst -*-

Recursive Feature Elimination
=============================

 =================   =====================
 Visualizer           :class:`~yellowbrick.model_selection.rfecv.RFECV`
 Quick Method         :func:`~yellowbrick.model_selection.rfecv.rfecv`
 Models               Classification, Regression
 Workflow             Model Selection
 =================   =====================

Recursive feature elimination (RFE) is a feature selection method that fits a model and removes the weakest feature (or features) until the specified number of features is reached. Features are ranked by the model's ``coef_`` or ``feature_importances_`` attributes, and by recursively eliminating a small number of features per loop, RFE attempts to eliminate dependencies and collinearity that may exist in the model.

RFE requires a specified number of features to keep, however it is often not known in advance how many features are valid. To find the optimal number of features cross-validation is used with RFE to score different feature subsets and select the best scoring collection of features. The ``RFECV`` visualizer plots the number of features in the model along with their cross-validated test score and variability and visualizes the selected number of features.

To show how this works in practice, we'll start with a contrived example using a dataset that has only 3 informative features out of 25.

.. note to contributors: the below code takes a long time to run so has not been
   modified with a plot directive. See rfecv.py to regenerate images.

.. code:: python

    from sklearn.svm import SVC
    from sklearn.datasets import make_classification

    from yellowbrick.model_selection import RFECV

    # Create a dataset with only 3 informative features
    X, y = make_classification(
        n_samples=1000, n_features=25, n_informative=3, n_redundant=2,
        n_repeated=0, n_classes=8, n_clusters_per_class=1, random_state=0
    )

    # Instantiate RFECV visualizer with a linear SVM classifier
    visualizer = RFECV(SVC(kernel='linear', C=1))

    visualizer.fit(X, y)        # Fit the data to the visualizer
    visualizer.show()           # Finalize and render the figure

.. image:: images/rfecv_sklearn_example.png

This figure shows an ideal RFECV curve, the curve jumps to an excellent accuracy when the three informative features are captured, then gradually decreases in accuracy as the non informative features are added into the model. The shaded area represents the variability of cross-validation, one standard deviation above and below the mean accuracy score drawn by the curve.

Exploring a real dataset, we can see the impact of RFECV on a credit default binary classifier.

.. note to contributors: the below code takes a long time to run so has not been
   modified with a plot directive. See rfecv.py to regenerate images.

.. code:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    from yellowbrick.model_selection import RFECV
    from yellowbrick.datasets import load_credit

    # Load classification dataset
    X, y = load_credit()

    cv = StratifiedKFold(5)
    visualizer = RFECV(RandomForestClassifier(), cv=cv, scoring='f1_weighted')

    visualizer.fit(X, y)        # Fit the data to the visualizer
    visualizer.show()           # Finalize and render the figure

.. image:: images/rfecv_credit.png

In this example we can see that 19 features were selected, though there doesn't appear to be much improvement in the f1 score of the model after around 5 features. Selection of the features to eliminate plays a large role in determining the outcome of each recursion; modifying the ``step`` parameter to eliminate more than one feature at each step may help to eliminate the worst features early, strengthening the remaining features (and can also be used to speed up feature elimination for datasets with a large number of features).

.. seealso:: This visualizer is is based off of the visualization in the scikit-learn documentation: `recursive feature elimination with cross-validation <http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html>`_. However, the Yellowbrick version does not use ``sklearn.feature_selection.RFECV`` but instead wraps ``sklearn.feature_selection.RFE`` models. The fitted model can be accessed on the visualizer using the ``viz.rfe_estimator_`` attribute, and in fact the visualizer acts as the fitted model when using ``predict()`` or ``score()``.

Quick Method
------------

The same functionality above can be achieved with the associated quick method ``rfecv``. This method will build the ``RFECV`` object with the associated arguments, fit it, then (optionally) immediately show the visualization.

.. code:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    from yellowbrick.model_selection import rfecv
    from yellowbrick.datasets import load_credit

    # Load classification dataset
    X, y = load_credit()

    cv = StratifiedKFold(5)
    visualizer = rfecv(RandomForestClassifier(), X=X, y=y, cv=cv, scoring='f1_weighted')

.. image:: images/rfecv_quick_method.png

API Reference
-------------

.. automodule:: yellowbrick.model_selection.rfecv
    :members: RFECV, rfecv
    :undoc-members:
    :show-inheritance:
