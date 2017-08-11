.. -*- mode: rst -*-

Visualizers and API
===================

Welcome the API documentation for Yellowbrick! This section contains a complete listing of all currently available, production-ready visualizers along with code examples of how to use them. Use the links below to navigate to the reference for each visualization.

.. toctree::
   :maxdepth: 2

   datasets
   anscombe
   features/index
   regressor/index
   classifier/index
   cluster/index
   text/index
   palettes

.. note:: Many examples utilize data from the UCI Machine Learning repository, in order to run the examples, make sure you follow the instructions in :doc:`datasets` to download and load required data.

A guide to finding the visualizer you're looking for: generally speaking, visualizers can be data visualizers which visualize instances relative to the model space; score visualizers which visualize model performance; model selection visualizers which compare multiple model forms against each other; and application specific-visualizers. This can be a bit confusing, so we've grouped visualizers according to the type of analysis they are well suited for.

Feature analysis visualizers are where you'll find the primary implementation of data visualizers. Regression, classification, and clustering analysis visualizers can be found in their respective libraries. Finally visualizers for text analysis are also available in Yellowbrick! Other utilities like styles, best fit lines, and anscombe's visualization can also be found in the links above.
