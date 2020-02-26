.. -*- mode: rst -*-

Silhouette Visualizer
=====================

The Silhouette Coefficient is used when the ground-truth about the dataset is unknown and computes the density of clusters computed by the model. The score is computed by averaging the silhouette coefficient for each sample, computed as the difference between the average intra-cluster distance and the mean nearest-cluster distance for each sample, normalized by the maximum value. This produces a score between 1 and -1, where 1 is highly dense clusters and -1 is completely incorrect clustering.

The Silhouette Visualizer displays the silhouette coefficient for each sample on a per-cluster basis, visualizing which clusters are dense and which are not. This is particularly useful for determining cluster imbalance, or for selecting a value for :math:`K` by comparing multiple visualizers.

=================   ==============================
Visualizer           :class:`~yellowbrick.cluster.silhouette.SilhouetteVisualizer`
Quick Method         :func:`~yellowbrick.cluster.silhouette.silhouette_visualizer`
Models               Clustering
Workflow             Model evaluation
=================   ==============================

Examples and demo

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
    visualizer.show()        # Finalize and render the figure


Quick Method
------------

The same functionality above can be achieved with the associated quick method `silhouette_visualizer`. This method will build the Silhouette Visualizer object with the associated arguments, fit it, then (optionally) immediately show it.

.. plot::
    :context: close-figs
    :alt: silhouette_visualizer on the nfl dataset with 4 clusters

    from sklearn.cluster import KMeans

    from yellowbrick.cluster import silhouette_visualizer
    from yellowbrick.datasets import load_credit

    # Load a clustering dataset
    X, y = load_credit()

    # Specify rows to cluster: under 40 y/o and have either graduate or university education
    X = X[(X['age'] <= 40) & (X['edu'].isin([1,2]))]

    # Use the quick method and immediately show the figure
    silhouette_visualizer(KMeans(5, random_state=42), X, colors='yellowbrick')

API Reference
-------------

.. automodule:: yellowbrick.cluster.silhouette
    :members: SilhouetteVisualizer, silhouette_visualizer
    :undoc-members:
    :show-inheritance:
