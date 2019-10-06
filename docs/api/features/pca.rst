.. -*- mode: rst -*-

PCA Projection
==============

The PCA Decomposition visualizer utilizes principal component analysis to decompose high dimensional data into two or three dimensions so that each instance can be plotted in a scatter plot. The use of PCA means that the projected dataset can be analyzed along axes of principal variation and can be interpreted to determine if spherical distance metrics can be utilized.

.. plot::
    :context: close-figs
    :alt: PCA Projection, 2D

    from yellowbrick.datasets import load_credit
    from yellowbrick.features.pca import PCADecomposition

    # Specify the features of interest and the target
    X, y = load_credit()

    # Create a list of colors to assign to points in the plot
    colors = np.array(['r' if yi else 'b' for yi in y])

    visualizer = PCADecomposition(scale=True, color=colors)
    visualizer.fit_transform(X, y)
    visualizer.show()


The PCA projection can also be plotted in three dimensions to attempt to visualize more principal components and get a better sense of the distribution in high dimensions.

.. plot::
    :context: close-figs
    :alt: PCA Projection, 3D

    from yellowbrick.datasets import load_credit
    from yellowbrick.features.pca import PCADecomposition

    X, y = load_credit()

    colors = np.array(['r' if yi else 'b' for yi in y])

    visualizer = PCADecomposition(scale=True, color=colors, proj_dim=3)
    visualizer.fit_transform(X, y)
    visualizer.show()


Biplot
------

The PCA projection can be enhanced to a biplot whose points are the projected instances and whose vectors represent the structure of the data in high dimensional space. By using ``proj_features=True``, vectors for each feature in the dataset are drawn on the scatter plot in the direction of the maximum variance for that feature. These structures can be used to analyze the importance of a feature to the decomposition or to find features of related variance for further analysis.

.. plot::
    :context: close-figs
    :alt: PCA biplot projection, 2D

    from yellowbrick.datasets import load_concrete
    from yellowbrick.features.pca import PCADecomposition

    # Load the concrete dataset
    X, y = load_concrete()

    visualizer = PCADecomposition(scale=True, proj_features=True)
    visualizer.fit_transform(X, y)
    visualizer.show()


.. plot::
    :context: close-figs
    :alt: PCA biplot projection, 3D

    from yellowbrick.datasets import load_concrete
    from yellowbrick.features.pca import PCADecomposition

    X, y = load_concrete()

    visualizer = PCADecomposition(scale=True, proj_features=True, proj_dim=3)
    visualizer.fit_transform(X, y)
    visualizer.show()

API Reference
-------------

.. automodule:: yellowbrick.features.pca
    :members: PCA
    :undoc-members:
    :show-inheritance:
