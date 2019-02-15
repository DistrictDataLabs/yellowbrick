.. -*- mode: rst -*-

Silhouette Visualizer
=====================

The Silhouette Coefficient is used when the ground-truth about the dataset is unknown and computes the density of clusters computed by the model. The score is computed by averaging the silhouette coefficient for each sample, computed as the difference between the average intra-cluster distance and the mean nearest-cluster distance for each sample, normalized by the maximum value. This produces a score between 1 and -1, where 1 is highly dense clusters and -1 is completely incorrect clustering.

The Silhouette Visualizer displays the silhouette coefficient for each sample on a per-cluster basis, visualizing which clusters are dense and which are not. This is particularly useful for determining cluster imbalance, or for selecting a value for :math:`K` by comparing multiple visualizers.

.. plot::
    :context: close-figs
    :alt: SilhouetteVisualizer on synthetic dataset with 8 random clusters

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.datasets import make_blobs

    from yellowbrick.cluster import SilhouetteVisualizer

    # Generate synthetic dataset with 8 random clusters
    X, y = make_blobs(n_samples=1000, n_features=12, centers=8, random_state=42)

    # Instantiate the clustering model and visualizer
    model = MiniBatchKMeans(6)
    visualizer = SilhouetteVisualizer(model)

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.poof()        # Draw/show/poof the data


API Reference
-------------

.. automodule:: yellowbrick.cluster.silhouette
    :members: SilhouetteVisualizer
    :undoc-members:
    :show-inheritance:
