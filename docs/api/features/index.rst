.. -*- mode: rst -*-

Feature Analysis Visualizers
============================

Feature analysis visualizers are designed to visualize instances in data
space in order to detect features or targets that might impact
downstream fitting. Because ML operates on high-dimensional data sets
(usually at least 35), the visualizers focus on aggregation,
optimization, and other techniques to give overviews of the data. It is
our intent that the steering process will allow the data scientist to
zoom and filter and explore the relationships between their instances
and between dimensions.

At the moment we have the following feature analysis visualizers implemented:

-  :doc:`rankd`: rank single and pairs of features to detect covariance
-  :doc:`radviz`: plot data points along axes ordered around a circle to detect separability
-  :doc:`pcoords`: plot instances as lines along vertical axes to
   detect classes or clusters
-  :doc:`pca`: project higher dimensions into a visual space using PCA
-  :doc:`manifold`: visualize high dimensional data using manifold learning
-  :doc:`importances`: rank features by relative importance in a model
-  :doc:`rfecv`: select a subset of features by importance
-  :doc:`jointplot`: (aka Jointplots) plot 2D correlation between features and target

Feature analysis visualizers implement the ``Transformer`` API from
scikit-learn, meaning they can be used as intermediate transform steps
in a ``Pipeline`` (particularly a ``VisualPipeline``). They are
instantiated in the same way, and then fit and transform are called on
them, which draws the instances correctly. Finally ``poof`` or ``show``
is called which displays the image.

.. code:: python

    # Feature Analysis Imports
    # NOTE that all these are available for import directly from the ``yellowbrick.features`` module
    from yellowbrick.features.rankd import Rank1D, Rank2D
    from yellowbrick.features.radviz import RadViz
    from yellowbrick.features.pcoords import ParallelCoordinates
    from yellowbrick.features.jointplot import JointPlotVisualizer
    from yellowbrick.features.pca import PCADecomposition
    from yellowbrick.features.manifold import Manifold
    from yellowbrick.features.importances import FeatureImportances
    from yellowbrick.features.rfecv import RFECV


.. toctree::
   :maxdepth: 2

   radviz
   rankd
   pcoords
   pca
   manifold
   importances
   rfecv
   jointplot
