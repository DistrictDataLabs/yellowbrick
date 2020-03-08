.. -*- mode: rst -*-

Elbow Method
============

The ``KElbowVisualizer`` implements the "elbow" method to help data scientists select the optimal number of clusters by fitting the model with a range of values for :math:`K`. If the line chart resembles an arm, then the "elbow" (the point of inflection on the curve) is a good indication that the underlying model fits best at that point. In the visualizer "elbow" will be annotated with a dashed line.

To demonstrate, in the following example the ``KElbowVisualizer`` fits the ``KMeans`` model for a range of :math:`K` values from 4 to 11 on a sample two-dimensional dataset with 8 random clusters of points. When the model is fit with 8 clusters, we can see a line annotating the "elbow" in the graph, which in this case we know to be the optimal number.

=================   ==============================
Visualizer           :class:`~yellowbrick.cluster.elbow.KElbowVisualizer`
Quick Method         :func:`~yellowbrick.cluster.elbow.kelbow_visualizer`
Models               Clustering
Workflow             Model evaluation
=================   ==============================

.. plot::
    :context: close-figs
    :alt: KElbowVisualizer on synthetic dataset with 8 random clusters

    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    from yellowbrick.cluster import KElbowVisualizer

    # Generate synthetic dataset with 8 random clusters
    X, y = make_blobs(n_samples=1000, n_features=12, centers=8, random_state=42)

    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(4,12))

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure

By default, the scoring parameter ``metric`` is set to ``distortion``, which
computes the sum of squared distances from each point to its assigned center.
However, two other metrics can also be used with the ``KElbowVisualizer`` -- ``silhouette`` and ``calinski_harabasz``. The ``silhouette`` score calculates the mean Silhouette Coefficient of all samples, while the ``calinski_harabasz`` score computes the ratio of dispersion between and within clusters.

The ``KElbowVisualizer`` also displays the amount of time to train the clustering model per :math:`K` as a dashed green line, but is can be hidden by setting ``timings=False``. In the following example, we'll use the ``calinski_harabasz`` score and hide the time to fit the model.

.. plot::
    :context: close-figs
    :alt: KElbowVisualizer on synthetic dataset with 8 random clusters

    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    from yellowbrick.cluster import KElbowVisualizer

    # Generate synthetic dataset with 8 random clusters
    X, y = make_blobs(n_samples=1000, n_features=12, centers=8, random_state=42)

    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(
        model, k=(4,12), metric='calinski_harabasz', timings=False
    )

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure

By default, the parameter ``locate_elbow`` is set to ``True``, which automatically find the "elbow" which likely corresponds to the optimal value of k using the "knee point detection algorithm". However, users can turn off the feature by setting ``locate_elbow=False``. You can read about the implementation of this algorithm at "`Knee point detection in Python <https://github.com/arvkevi/kneed>`_" by Kevin Arvai.

In the following example, we'll use the ``calinski_harabasz`` score and turn off ``locate_elbow`` feature.

.. plot::
    :context: close-figs
    :alt: KElbowVisualizer on synthetic dataset with 8 random clusters

    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    from yellowbrick.cluster import KElbowVisualizer

    # Generate synthetic dataset with 8 random clusters
    X, y = make_blobs(n_samples=1000, n_features=12, centers=8, random_state=42)

    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(
        model, k=(4,12), metric='calinski_harabasz', timings=False, locate_elbow=False
    )

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure

It is important to remember that the "elbow" method does not work well if the data
is not very clustered. In this case, you might see a smooth curve and the optimal value of :math:`K` will be unclear.

Quick Method
------------

The same functionality above can be achieved with the associated quick method ``kelbow_visualizer``. This method will build the ``KElbowVisualizer`` object with the associated arguments, fit it, then (optionally) immediately show the visualization.

.. plot::
    :context: close-figs
    :alt: KElbowVisualizer

    from sklearn.cluster import KMeans
    from yellowbrick.cluster.elbow import kelbow_visualizer
    from yellowbrick.datasets.loaders import load_nfl

    X, y = load_nfl()

    # Use the quick method and immediately show the figure
    kelbow_visualizer(KMeans(random_state=4), X, k=(2,10))


API Reference
-------------

.. automodule:: yellowbrick.cluster.elbow
    :members: KElbowVisualizer, kelbow_visualizer
    :undoc-members:
    :show-inheritance:
