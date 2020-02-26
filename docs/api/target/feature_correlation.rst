.. -*- mode: rst -*-

Feature Correlation
===================

This visualizer calculates Pearson correlation coefficients and mutual information between features and the dependent variable.
This visualization can be used in feature selection to identify features with high correlation or large mutual information with the dependent variable.

Pearson Correlation
-------------------

The default calculation is Pearson correlation, which is performed with ``scipy.stats.pearsonr``.

=================   =================
Visualizer           :class:`~yellowbrick.target.feature_correlation.FeatureCorrelation`
Quick Method         :func:`~yellowbrick.target.feature_correlation.feature_correlation`
Models               Regression/Classification/Clustering
Workflow             Feature Engineering/Model Selection
=================   =================

.. plot::
    :context: close-figs
    :alt: FeatureCorrelation on the diabetes dataset using Pearson correlation

    from sklearn import datasets
    from yellowbrick.target import FeatureCorrelation

    # Load the regression dataset
    data = datasets.load_diabetes()
    X, y = data['data'], data['target']

    # Create a list of the feature names
    features = np.array(data['feature_names'])

    # Instantiate the visualizer
    visualizer = FeatureCorrelation(labels=features)

    visualizer.fit(X, y)        # Fit the data to the visualizer
    visualizer.show()           # Finalize and render the figure


Mutual Information - Regression
-------------------------------

Mutual information between features and the dependent variable is calculated with ``sklearn.feature_selection.mutual_info_classif`` when ``method='mutual_info-classification'`` and ``mutual_info_regression`` when ``method='mutual_info-regression'``.
It is very important to specify discrete features when calculating mutual information because the calculation for continuous and discrete variables are different.
See `scikit-learn documentation <http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html>`_ for more details.

.. plot::
    :context: close-figs
    :alt: FeatureCorrelation on the diabetes dataset using mutual_info-regression

    from sklearn import datasets
    from yellowbrick.target import FeatureCorrelation

    # Load the regression dataset
    data = datasets.load_diabetes()
    X, y = data['data'], data['target']

    # Create a list of the feature names
    features = np.array(data['feature_names'])

    # Create a list of the discrete features
    discrete = [False for _ in range(len(features))]
    discrete[1] = True

    # Instantiate the visualizer
    visualizer = FeatureCorrelation(method='mutual_info-regression', labels=features)

    visualizer.fit(X, y, discrete_features=discrete, random_state=0)
    visualizer.show()


Mutual Information - Classification
-----------------------------------

By fitting with a pandas DataFrame, the feature labels are automatically obtained from the column names.
This visualizer also allows sorting of the bar plot according to the calculated mutual information (or Pearson correlation coefficients) and selecting features to plot by specifying the names of the features or the feature index.

.. plot::
    :context: close-figs
    :alt: FeatureCorrelation on the wine dataset using mutual_info-classification

    import pandas as pd

    from sklearn import datasets
    from yellowbrick.target import FeatureCorrelation

    # Load the regression dataset
    data = datasets.load_wine()
    X, y = data['data'], data['target']
    X_pd = pd.DataFrame(X, columns=data['feature_names'])

    # Create a list of the features to plot
    features = ['alcohol', 'ash', 'hue', 'proline', 'total_phenols']

    # Instaniate the visualizer
    visualizer = FeatureCorrelation(
        method='mutual_info-classification', feature_names=features, sort=True
    )

    visualizer.fit(X_pd, y)        # Fit the data to the visualizer
    visualizer.show()              # Finalize and render the figure


Quick Method
------------

The same functionality above can be achieved with the associated quick method ``feature_correlation``. This method will build the ``FeatureCorrelation`` object with the associated arguments, fit it, then (optionally) immediately show it

.. plot::
    :context: close-figs
    :alt: feature_correlation on the diabetes dataset

    import numpy as np
    from sklearn import datasets
    import matplotlib.pyplot as plt
    from yellowbrick.target.feature_correlation import feature_correlation

    #Load the diabetes dataset
    data = datasets.load_iris()
    X, y = data['data'], data['target']

    features = np.array(data['feature_names'])
    visualizer = feature_correlation(X, y, labels=features)
    plt.tight_layout()


API Reference
-------------

.. automodule:: yellowbrick.target.feature_correlation
    :members: FeatureCorrelation, feature_correlation
    :undoc-members:
    :show-inheritance:
