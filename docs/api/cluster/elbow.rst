.. -*- mode: rst -*-

Elbow Method
============

The ``KElbowVisualizer`` implements the "elbow" method to help data scientists select the optimal number of clusters by fitting the model with a range of values for :math:`K`. If the line chart resembles an arm, then the "elbow" (the point of inflection on the curve) is a good indication that the underlying model fits best at that point.

To demonstrate, in the following example the ``KElbowVisualizer`` fits the ``KMeans`` model for a range of :math:`K` values from 4 to 11 on a sample two-dimensional dataset with 8 random clusters of points. When the model is fit with 8 clusters, we can see an "elbow" in the graph, which in this case we know to be the optimal number.

.. plot::

    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    from yellowbrick.cluster import KElbowVisualizer

    # Create synthetic dataset with 8 random clusters
    X, y = make_blobs(centers=8, n_features=12, shuffle=True, random_state=42)

    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(4,12))

    visualizer.fit(X)    # Fit the data to the visualizer
    visualizer.poof()    # Draw/show/poof the data

By default, the scoring parameter ``metric`` is set to ``distortion``, which
computes the sum of squared distances from each point to its assigned center.
However, two other metrics can also be used with the ``KElbowVisualizer`` -- ``silhouette`` and ``calinski_harabaz``. The ``silhouette`` score calculates the mean Silhouette Coefficient of all samples, while the ``calinski_harabaz`` score computes the ratio of dispersion between and within clusters.

The ``KElbowVisualizer`` also displays the amount of time to train the clustering model per :math:`K` as a dashed green line, but is can be hidden by setting ``timings=False``. In the following example, we'll use the ``calinski_harabaz`` score and hide the time to fit the model.

.. plot::

    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    from yellowbrick.cluster import KElbowVisualizer

    # Create synthetic dataset with 8 random clusters
    X, _ = make_blobs(centers=8, n_features=12, shuffle=True, random_state=42)

    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(4,12), metric='calinski_harabaz', timings=False)

    visualizer.fit(X)    # Fit the data to the visualizer
    visualizer.poof()    # Draw/show/poof the data

It is important to remember that the "elbow" method does not work well if the data
is not very clustered. In this case, you might see a smooth curve and the optimal value of :math:`K` will be unclear.

API Reference
-------------

.. automodule:: yellowbrick.cluster.elbow
    :members: KElbowVisualizer
    :undoc-members:
    :show-inheritance:
