.. -*- mode: rst -*-

PCA Projection
==============

The PCA Decomposition visualizer utilizes principal component analysis to decompose high dimensional data into two or three dimensions so that each instance can be plotted in a scatter plot. The use of PCA means that the projected dataset can be analyzed along axes of principal variation and can be interpreted to determine if spherical distance metrics can be utilized.

=================   =================
Visualizer            `PCA Decomposition <https://www.scikit-yb.org/en/latest/api/features/pca.html#yellowbrick.features.pca.PCA>`_
Quick Method          `pca_decomposition <https://www.scikit-yb.org/en/latest/api/features/pca.html#yellowbrick.features.pca.pca_decomposition>`_
Models               Classification
Workflow             Feature extraction
=================   =================

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

Quick Method
----------------------
The same functionality above can be achieved with the associated quick method ``pca_decomposition``. This method
will build the ``PCADecomposition`` object with the associated arguments, fit it, then (optionally) immediately
show it.

.. plot::
    :context: close-figs
    :alt: pca_decomposition on the credit dataset

    from yellowbrick.datasets import load_credit
    from yellowbrick.features.pca import pca_decomposition

    # Specify the features of interest and the target
    X, y = load_credit()

    # Create a list of colors to assign to points in the plot
    colors = np.array(['r' if yi else 'b' for yi in y])

    # Instantiate the visualizer
    visualizer = pca_decomposition(X, y, scale=True, color=colors)

API Reference
-------------

.. automodule:: yellowbrick.features.pca
    :members: PCA, pca_decomposition
    :undoc-members:
    :show-inheritance:
