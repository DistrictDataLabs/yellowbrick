.. -*- mode: rst -*-

Intercluster Distance Maps
==========================

Intercluster distance maps display an embedding of the cluster centers in 2 dimensions with the distance to other centers preserved. E.g. the closer to centers are in the visualization, the closer they are in the original feature space. The clusters are sized according to a scoring metric. By default, they are sized by membership, e.g. the number of instances that belong to each center. This gives a sense of the relative importance of clusters. Note however, that because two clusters overlap in the 2D space, it does not imply that they overlap in the original feature space.

.. plot::
    :context: close-figs
    :alt: Intercluster Distance Visualizer on dataset with 12 random clusters

    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    from yellowbrick.cluster import InterclusterDistance

    # Generate synthetic dataset with 12 random clusters
    X, y = make_blobs(n_samples=1000, n_features=12, centers=12, random_state=42)

    # Instantiate the clustering model and visualizer
    model = KMeans(6)
    visualizer = InterclusterDistance(model)

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure

API Reference
-------------

.. automodule:: yellowbrick.cluster.icdm
    :members: InterclusterDistance
    :undoc-members:
    :show-inheritance:
