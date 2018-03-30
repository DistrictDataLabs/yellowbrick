.. -*- mode: rst -*-

Model Selection Visualizers
===========================

Yellowbrick visualizers are intended to steer the model selection process. Generally, model selection is a search problem defined as follows: given N instances described by numeric properties and (optionally) a target for estimation, find a model described by a triple composed of features, an algorithm and hyperparameters that best fits the data. For most purposes the "best" triple refers to the triple that receives the best cross-validated score for the model type.

The ``yellowbrick.model_selection`` package provides visualizers for inspecting the performance of cross validation and hyper parameter tuning. Many visualizers wrap functionality found in ``sklearn.model_selection`` and others build upon it for performing multi-model comparisons.

The currently implemented model selection visualizers are as follows:

.. toctree::
   :maxdepth: 2
