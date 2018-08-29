.. -*- mode: rst -*-

Target Visualizers
==================

Target visualizers specialize in visually describing the dependent variable for supervised modeling, often referred to as ``y`` or the target.

The following visualizations are currently implemented:

-  :doc:`class_balance`: Visual inspection of the target to show the support of each class to the final estimator.
-  :doc:`feature_correlation`: Plot correlation between features and dependent variables.

.. code:: python

    # Target Visualizers Imports
    from yellowbrick.classifier import FeatureCorrelation, ClassBalance

.. toctree::
   :maxdepth: 2

   class_balance
   feature_correlation
