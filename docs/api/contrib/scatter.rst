.. -*- mode: rst -*-

Scatter Plot Visualizer
=======================

Sometimes for feature analysis you simply need a scatter plot to determine the distribution of data. Machine learning operates on high dimensional data, so the number of dimensions has to be filtered. As a result these visualizations are typically used as the base for larger visualizers; however you can also use them to quickly plot data during ML analysis.

A scatter visualizer simply plots two features against each other and colors the points according to the target. This can be useful in assessing the relationship of pairs of features to an individual target.

.. code:: python

    # Load the classification data set
    data = load_data("occupancy")

    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    classes = ["unoccupied", "occupied"]

    # Extract the numpy arrays from the data frame
    X = data[features]
    y = data.occupancy

.. code:: python

    from yellowbrick.contrib.scatter import ScatterVisualizer

    visualizer = ScatterVisualizer(x="light", y="C02", classes=classes)

    visualizer.fit(X, y)
    visualizer.transform(X)
    visualizer.poof()


.. image:: images/scatter.png


API Reference
-------------

.. automodule:: yellowbrick.contrib.scatter
    :members: ScatterVisualizer
    :undoc-members:
    :show-inheritance:
