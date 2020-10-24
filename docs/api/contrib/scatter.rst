.. -*- mode: rst -*-

Scatter Plot Visualizer
=======================

Sometimes for feature analysis you simply need a scatter plot to determine the distribution of data. Machine learning operates on high dimensional data, so the number of dimensions has to be filtered. As a result these visualizations are typically used as the base for larger visualizers; however you can also use them to quickly plot data during ML analysis.

A scatter visualizer simply plots two features against each other and colors the points according to the target. This can be useful in assessing the relationship of pairs of features to an individual target.

.. plot::
    :context: close-figs
    :alt: ScatterVisualizer on occupancy dataset

    from yellowbrick.contrib.scatter import ScatterVisualizer
    from yellowbrick.datasets import load_occupancy

    # Load the classification dataset
    X, y = load_occupancy()

    # Specify the target classes
    classes = ["unoccupied", "occupied"]

    # Instantiate the visualizer
    visualizer = ScatterVisualizer(x="light", y="CO2", classes=classes)

    visualizer.fit(X, y)           # Fit the data to the visualizer
    visualizer.transform(X)        # Transform the data
    visualizer.show()              # Finalize and render the figure


API Reference
-------------

.. automodule:: yellowbrick.contrib.scatter
    :members: ScatterVisualizer
    :undoc-members:
    :show-inheritance:
