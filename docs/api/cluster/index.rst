.. -*- mode: rst -*-

Clustering Visualizers
======================

Clustering models are unsupervised methods that attempt to detect patterns in unlabeled data. There are two primary classes of clustering algorithm: *agglomerative* clustering links similar data points together, whereas *centroidal* clustering attempts to find centers or partitions in the data. Yellowbrick provides the `yellowbrick.cluster` module to visualize and evaluate clustering behavior. Currently we provide two visualizers to evaluate *centroidal* mechanisms, particularly K-Means clustering, that help us to discover an optimal :math:`K` parameter in the clustering metric:

-  :doc:`elbow`: visualize the clusters according to some scoring function, look for an "elbow" in the curve.
-  :doc:`silhouette`: visualize the silhouette scores of each cluster in a single model.

Because it is very difficult to `score` a clustering model, Yellowbrick visualizers wrap Scikit-Learn "clusterer" estimators via their `fit()` method. Once the clustering model is trained, then the visualizer can call `poof()` to display the clustering evaluation metric.

.. toctree::
   :maxdepth: 2

   elbow
   silhouette
