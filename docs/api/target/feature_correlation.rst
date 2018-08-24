.. -*- mode: rst -*-

Feature Correlation
===================

This visualizer calculates Pearson correlation coefficients and mutual information between features and the dependent variable.
This visualization can be used in feature selection to identify features with high correlation or large mutual information with the dependent variable.

Pearson Correlation
-------------------

The default calculation is Pearson correlation, which is perform with ``scipy.stats.pearsonr``.

.. code:: python

    from sklearn import datasets
    from yellowbrick.target import FeatureCorrelation

    # Load the regression data set
    data = datasets.load_diabetes()
    X, y = data['data'], data['target']
    feature_names = np.array(data['feature_names'])

    visualizer = FeatureCorrelation(labels=feature_names)
    visualizer.fit(X, y)
    visualizer.poof()

.. image:: images/feature_correlation_pearson.png

Mutual Information - Regression
-------------------------------

Mutual information between features and the dependent variable is calculated with ``sklearn.feature_selection.mutual_info_classif`` when ``method='mutual_info-classification'`` and ``mutual_info_regression`` when ``method='mutual_info-regression'``.
It is very important to specify discrete features when calculating mutual information because the calculation for continuous and discrete variables are different.
See `scikit-learn documentation <http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html>`_ for more details.

.. code:: python

    from sklearn import datasets
    from yellowbrick.target import FeatureCorrelation

    # Load the regression data set
    data = datasets.load_diabetes()
    X, y = data['data'], data['target']
    feature_names = np.array(data['feature_names'])

    discrete_features = [False for _ in range(len(feature_names))]
    discrete_features[1] = True

    visualizer = FeatureCorrelation(method='mutual_info-regression',
                                    labels=feature_names)
    visualizer.fit(X, y, discrete_features=discrete_features, random_state=0)
    visualizer.poof()

.. image:: images/feature_correlation_mutual_info_regression.png

Mutual Information - Classification
-----------------------------------

By fitting with a pandas DataFrame, the feature labels are automatically obtained from the column names.
This visualizer also allows sorting of the bar plot according to the calculated mutual information (or Pearson correlation coefficients) and selecting features to plot by specifying the names of the features or the feature index.

.. code:: python

    from sklearn import datasets
    from yellowbrick.target import FeatureCorrelation

    # Load the regression data set
    data = datasets.load_diabetes()
    X, y = data['data'], data['target']
    feature_names = np.array(data['feature_names'])
    X_pd = pd.DataFrame(X, columns=feature_names)

    feature_to_plot = ['alcohol', 'ash', 'hue', 'proline', 'total_phenols']

    visualizer = FeatureCorrelation(method='mutual_info-classification',
                                    feature_names=feature_to_plot, sort=True)
    visualizer.fit(X_pd, y, random_state=0)
    visualizer.poof()

.. image:: images/feature_correlation_mutual_info_classification.png

API Reference
-------------

.. automodule:: yellowbrick.target.feature_correlation
    :members: FeatureCorrelation
    :undoc-members:
    :show-inheritance:
