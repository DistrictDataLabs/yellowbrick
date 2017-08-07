Parallel Coordinates
====================

Parallel coordinates displays each feature as a vertical axis spaced
evenly along the horizontal, and each instance as a line drawn between
each individual axis. This allows many dimensions; in fact given
infinite horizontal space (e.g. a scrollbar), an infinite number of
dimensions can be displayed!

Data scientists use this method to detect clusters of instances that
have similar classes, and to note features that have high variance or
different distributions.

.. code:: python

    # Load the classification data set
    data = load_data('occupancy')

    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    classes = ['unoccupied', 'occupied']

    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.occupancy.as_matrix()

.. code:: python

    # Instantiate the visualizer
    visualizer = ParallelCoordinates(classes=classes, features=features)

    visualizer.fit(X, y)      # Fit the data to the visualizer
    visualizer.transform(X)   # Transform the data
    visualizer.poof()         # Draw/show/poof the data


.. image:: images/parallel_coordinates.png

API Reference
-------------

.. automodule:: yellowbrick.features.pcoords
    :members: ParallelCoordinates
    :undoc-members:
    :show-inheritance:
