.. -*- mode: rst -*-

Silhouette Visualizer
=====================

The Silhouette Coefficient is used when the ground-truth about the dataset is unknown and computes the density of clusters computed by the model. The score is computed by averaging the silhouette coefficient for each sample, computed as the difference between the average intra-cluster distance and the mean nearest-cluster distance for each sample, normalized by the maximum value. This produces a score between 1 and -1, where 1 is highly dense clusters and -1 is completely incorrect clustering.

The Silhouette Visualizer displays the silhouette coefficient for each sample on a per-cluster basis, visualizing which clusters are dense and which are not. This is particularly useful for determining cluster imbalance, or for selecting a value for :math:`K` by comparing multiple visualizers.

.. plot::
    :context: close-figs
    :alt: SilhouetteVisualizer on the nfl dataset with 4 clusters

    from sklearn.cluster import KMeans

    from yellowbrick.cluster import SilhouetteVisualizer
    from yellowbrick.datasets import load_nfl

    # Load a clustering dataset
    X, y = load_nfl()

    # Specify the features to use for clustering
    features = ['Rec', 'Yds', 'TD', 'Fmb', 'Ctch_Rate']
    X = X.query('Tgt >= 20')[features]

    # Instantiate the clustering model and visualizer
    model = KMeans(5, random_state=42)
    visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.poof()        # Draw/show/poof the data


API Reference
-------------

.. automodule:: yellowbrick.cluster.silhouette
    :members: SilhouetteVisualizer
    :undoc-members:
    :show-inheritance:
