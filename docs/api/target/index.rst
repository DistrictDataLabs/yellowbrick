.. -*- mode: rst -*-

Target Visualizers
==================

Target visualizers specialize in visually describing the dependent variable for supervised modeling, often referred to as ``y`` or the target.

The following visualizations are currently implemented:

-  :doc:`binning`: Generate histogram with vertical lines showing the recommended value point to bin data into evenly distributed bins.
-  :doc:`class_balance`: Visual inspection of the target to show the support of each class to the final estimator.
-  :doc:`feature_correlation`: Plot correlation between features and dependent variables.

.. code:: python

    # Target Visualizers Imports
    from yellowbrick.target import BalancedBinningReference
    from yellowbrick.target import ClassBalance
    from yellowbrick.target import FeatureCorrelation

.. toctree::
   :maxdepth: 2

   binning
   class_balance
   feature_correlation
