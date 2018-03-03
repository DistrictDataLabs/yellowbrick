.. -*- mode: rst -*-

|Visualizers|_

.. |Visualizers| image:: http://www.scikit-yb.org/en/latest/_images/visualizers.png
    :width: 800 px
.. _Visualizers: http://www.scikit-yb.org/

Yellowbrick
===========

Yellowbrick is a suite of visual analysis and diagnostic tools designed to facilitate machine learning with Scikit-Learn. The library implements a new core API object, the "Visualizer" that is an Scikit-Learn estimator: an object that learns from data. Like transformers or models, visualizers learn from data by creating a visual representation of the model selection workflow.

Visualizers allow users to steer the model selection process, building intuition around feature engineering, algorithm selection, and hyperparameter tuning. For example, visualizers can help diagnose common problems surrounding model complexity and bias, heteroscedasticity, underfit and overtraining, or class balance issues. By applying visualizers to the model selection workflow, Yellowbrick allows you to steer predictive models to more successful results, faster.

Please see the full documentation at: http://scikit-yb.org/ particularly the `quick start guide <http://www.scikit-yb.org/en/latest/quickstart.html>`_

Visualizers
-----------

Visualizers are estimators (objects that learn from data) whose primary objective is to create visualizations that allow insight into the model selection process. In Scikit-Learn terms, they can be similar to transformers when visualizing the data space or wrap an model estimator similar to how the “ModelCV” (e.g. RidgeCV_, LassoCV_) methods work. The primary goal of Yellowbrick is to create a sensical API similar to Scikit-Learn. Some of our most popular visualizers include:

.. _RidgeCV: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
.. _LassoCV: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html

Feature Visualization
~~~~~~~~~~~~~~~~~~~~~

- **Rank Features**: single or pairwise ranking of features to detect relationships
- **Parallel Coordinates**: horizontal visualization of instances
- **Radial Visualization**: separation of instances around a circular plot
- **PCA Projection**: projection of instances based on principal components
- **Feature Importances**: rank features based on their in-model performance
- **Scatter and Joint Plots**: direct data visualization with feature selection

Classification Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Class Balance**: see how the distribution of classes affects the model
- **Classification Report**: visual representation of precision, recall, and F1
- **ROC/AUC Curves**: receiver operator characteristics and area under the curve
- **Confusion Matrices**: visual description of class decision making

Regression Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

- **Prediction Error Plots**: find model breakdowns along the domain of the target
- **Residuals Plot**: show the difference in residuals of training and test data
- **Alpha Selection**: show how the choice of alpha influences regularization

Clustering Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

- **K-Elbow Plot**: select k using the elbow method and various metrics
- **Silhouette Plot**: select k by visualizing silhouette coefficient values

Text Visualization
~~~~~~~~~~~~~~~~~~

- **Term Frequency**: visualize the frequency distribution of terms in the corpus
- **TSNE**: use stochastic neighbor embedding to project documents.

... and more! Visualizers are being added all the time; be sure to check the examples_ (or even the develop_ branch) and feel free to contribute your ideas for new Visualizers!

.. _examples: http://www.scikit-yb.org/en/latest/api/index.html
.. _develop: https://github.com/districtdatalabs/yellowbrick/tree/develop
