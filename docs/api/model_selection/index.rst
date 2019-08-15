.. -*- mode: rst -*-

Model Selection Visualizers
===========================

Yellowbrick visualizers are intended to steer the model selection process. Generally, model selection is a search problem defined as follows: given N instances described by numeric properties and (optionally) a target for estimation, find a model described by a triple composed of features, an algorithm and hyperparameters that best fits the data. For most purposes the "best" triple refers to the triple that receives the best cross-validated score for the model type.

The ``yellowbrick.model_selection`` package provides visualizers for inspecting the performance of cross validation and hyper parameter tuning. Many visualizers wrap functionality found in ``sklearn.model_selection`` and others build upon it for performing multi-model comparisons.

The currently implemented model selection visualizers are as follows:

-  :doc:`validation_curve`: visualizes how the adjustment of a hyperparameter influences training and test scores to tune the bias/variance trade-off.
-  :doc:`learning_curve`: shows how the size of training data influences the model to diagnose if a model suffers more from variance error vs. bias error.
-  :doc:`cross_validation`: displays cross-validated scores as a bar chart with average as a horizontal line.
-  :doc:`importances`: rank features by relative importance in a model
-  :doc:`rfecv`: select a subset of features by importance

Model selection makes heavy use of cross validation to measure the performance of an estimator. Cross validation splits a dataset into a training data set and a test data set; the model is fit on the training data and evaluated on the test data. This helps avoid a common pitfall, overfitting, where the model simply memorizes the training data and does not generalize well to new or unknown input.

There are many ways to define how to split a dataset for cross validation. For more information on how scikit-learn implements these mechanisms, please review `Cross-validation: evaluating estimator performance <http://scikit-learn.org/stable/modules/cross_validation.html>`_ in the scikit-learn documentation.

.. toctree::
   :maxdepth: 2

   validation_curve
   learning_curve
   cross_validation
   importances
   rfecv
