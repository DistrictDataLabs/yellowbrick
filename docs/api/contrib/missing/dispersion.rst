.. -*- mode: rst -*-

MissingValues Dispersion
========================

The MissingValues Dispersion visualizer creates a chart that maps the position of missing values by the order of the index.


Without Targets Supplied
------------------------

.. plot::
    :context: close-figs
    :alt: MissingValues Dispersion visualization on a dataset with no targets supplied

    import numpy as np

    from sklearn.datasets import make_classification
    from yellowbrick.contrib.missing import MissingValuesDispersion

    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=2, n_redundant=3,
        n_classes=2, n_clusters_per_class=2, random_state=854
    )

    # assign some NaN values
    X[X > 1.5] = np.nan
    features = ["Feature {}".format(str(n)) for n in range(10)]

    visualizer = MissingValuesDispersion(features=features)

    visualizer.fit(X)
    visualizer.show()


With Targets (y) Supplied
-------------------------

.. plot::
    :context: close-figs
    :alt: MissingValues Dispersion visualization on a dataset with no targets supplied

    import numpy as np

    from sklearn.datasets import make_classification
    from yellowbrick.contrib.missing import MissingValuesDispersion

    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=2, n_redundant=3,
        n_classes=2, n_clusters_per_class=2, random_state=854
    )

    # assign some NaN values
    X[X > 1.5] = np.nan
    features = ["Feature {}".format(str(n)) for n in range(10)]

    # Instantiate the visualizer
    visualizer = MissingValuesDispersion(features=features)

    visualizer.fit(X, y=y) # supply the targets via y
    visualizer.show()


API Reference
-------------

.. automodule:: yellowbrick.contrib.missing.dispersion
    :members: MissingValuesDispersion
    :undoc-members:
    :show-inheritance:
