.. -*- mode: rst -*-

Intercluster Distance Maps
==========================

Intercluster distance maps display an embedding of the cluster centers in 2 dimensions with the distance to other centers preserved. E.g. the closer to centers are in the visualization, the closer they are in the original feature space. The clusters are sized according to a scoring metric. By default, they are sized by membership, e.g. the number of instances that belong to each center. This gives a sense of the relative importance of clusters. Note however, that because two clusters overlap in the 2D space, it does not imply that they overlap in the original feature space.

.. code:: python

    from sklearn.datasets import make_blobs

    # Make 12 blobs dataset
    X, y = make_blobs(centers=12, n_samples=1000, n_features=16, shuffle=True)

.. code:: python

    from sklearn.cluster import KMeans
    from yellowbrick.cluster import InterclusterDistance

    # Instantiate the clustering model and visualizer
    visualizer = InterclusterDistance(KMeans(9))

    visualizer.fit(X) # Fit the training data to the visualizer
    visualizer.poof() # Draw/show/poof the data


.. image:: images/icdm.png

API Reference
-------------

.. automodule:: yellowbrick.cluster.icdm
    :members: InterclusterDistance
    :undoc-members:
    :show-inheritance:
