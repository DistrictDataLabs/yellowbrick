.. -*- mode: rst -*-

MissingValues Bar
=================

The MissingValues Bar visualizer creates a bar graph that counts the number of missing values per feature column. If the target ``y`` is supplied to fit, a stacked bar chart is produced.


Without Targets Supplied
------------------------

.. plot::
    :context: close-figs
    :alt: MissingValues Bar visualization on a dataset with no targets supplied

    import numpy as np

    from sklearn.datasets import make_classification
    from yellowbrick.contrib.missing import MissingValuesBar

    # Make a classification dataset
    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=2, n_redundant=3,
        n_classes=2, n_clusters_per_class=2, random_state=854
    )

    # Assign NaN values
    X[X > 1.5] = np.nan
    features = ["Feature {}".format(str(n)) for n in range(10)]

    # Instantiate the visualizer
    visualizer = MissingValuesBar(features=features)

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure


With Targets (``y``) Supplied
-----------------------------

.. plot::
    :context: close-figs
    :alt: MissingValuesBar visualization on a dataset with targets supplied

    import numpy as np

    from sklearn.datasets import make_classification
    from yellowbrick.contrib.missing import MissingValuesBar

    # Make a classification dataset
    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=2, n_redundant=3,
        n_classes=2, n_clusters_per_class=2, random_state=854
    )

    # Assign NaN values
    X[X > 1.5] = np.nan
    features = ["Feature {}".format(str(n)) for n in range(10)]

    # Instantiate the visualizer
    visualizer = MissingValuesBar(features=features)

    visualizer.fit(X, y=y)        # Supply the targets via y
    visualizer.show()             # Finalize and render the figure


API Reference
-------------

.. automodule:: yellowbrick.contrib.missing.bar
    :members: MissingValuesBar
    :undoc-members:
    :show-inheritance:
