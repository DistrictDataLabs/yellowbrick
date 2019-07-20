.. -*- mode: rst -*-

Direct Data Visualization
=========================

Sometimes for feature analysis you simply need a scatter plot to determine the distribution of data. Machine learning operates on high dimensional data, so the number of dimensions has to be filtered. As a result these visualizations are typically used as the base for larger visualizers; however you can also use them to quickly plot data during ML analysis.

Joint Plot Visualization
------------------------

A joint plot visualizer plots a feature against the target and shows the distribution of each via a histogram on each axis.

.. code:: python

    from yellowbrick.datasets import load_concrete
    from yellowbrick.features import JointPlotVisualizer

    # Load the dataset
    X, y = load_concrete()

    # Select the feature and target from the dataset
    X = X["cement"]
    y = y.values

    # Instantiate the visualizer
    visualizer = JointPlotVisualizer(feature="cement", target="strength")

    visualizer.fit(X, y)        # Fit the data to the visualizer
    visualizer.poof()           # Draw/show/poof the data


.. image:: images/jointplot.png

The joint plot visualizer can also be plotted with hexbins in the case of many, many points.

.. code:: python

    from yellowbrick.datasets import load_concrete
    from yellowbrick.features import JointPlotVisualizer

    # Load the dataset
    X, y = load_concrete()

    # Select the feature and target from the dataset
    X = X["cement"]
    y = y.values

    # Instantiate the visualizer
    visualizer = JointPlotVisualizer(
        feature="cement", target="strength", joint_plot='hex'
    )

    visualizer.fit(X, y)        # Fit the data to the visualizer
    visualizer.poof()           # Draw/show/poof the data

.. image:: images/jointplot_hex.png


API Reference
-------------

.. automodule:: yellowbrick.features.jointplot
    :members: JointPlot
    :undoc-members:
    :show-inheritance:
