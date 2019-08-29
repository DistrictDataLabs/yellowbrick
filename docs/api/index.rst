.. -*- mode: rst -*-

Visualizers and API
===================

Welcome to the API documentation for Yellowbrick! This section contains a complete listing of the currently available, production-ready visualizers along with code examples of how to use them. You may use the following links to navigate to the reference material for each visualization.

.. toctree::
   :maxdepth: 2

   datasets/index
   anscombe
   features/index
   target/index
   regressor/index
   classifier/index
   cluster/index
   model_selection/index
   text/index
   contrib/index
   palettes
   figures

.. note:: Many examples utilize data from the UCI Machine Learning repository. In order to run the accompanying code, make sure to follow the instructions in :doc:`datasets/index` to download and load the required data.

A guide to finding the visualizer you're looking for: generally speaking, visualizers can be data visualizers which visualize instances relative to the model space; score visualizers which visualize model performance; model selection visualizers which compare multiple model forms against each other; and application specific-visualizers. This can be a bit confusing, so we've grouped visualizers according to the type of analysis they are well suited for.

Feature analysis visualizers are where you'll find the primary implementation of data visualizers. Regression, classification, and clustering analysis visualizers can be found in their respective libraries. Finally, visualizers for text analysis are also available in Yellowbrick! Other utilities, such as styles, best fit lines, and Anscombe's visualization, can also be found in the links above.
