.. -*- mode: rst -*-

Parallel Coordinates
====================

Parallel coordinates is multi-dimensional feature visualization technique  where the vertical axis is duplicated horizontally for each feature. Instances are displayed as a single line segment drawn from each vertical axes to the location representing their value for that feature. This allows many dimensions to be visualized at once; in fact given infinite horizontal space (e.g. a scrolling window), technically an infinite number of dimensions can be displayed!

Data scientists use this method to detect clusters of instances that have similar classes, and to note features that have high variance or different distributions. We can see this in action after first loading our occupancy classification dataset. Note that the visualization can be drawn with either the ``ParallelCoordinates`` visualizer or using the ``parallel_coordinates`` quick method:

.. plot::

    from yellowbrick.features import ParallelCoordinates
    from yellowbrick.datasets import load_occupancy

    # Load the classification data set
    X, y = load_occupancy()

    # Specify the features of interest and the classes of the target
    features = [
        "temperature", "relative humidity", "light", "C02", "humidity"
    ]
    classes = ["unoccupied", "occupied"]

    # Instantiate the visualizer
    visualizer = ParallelCoordinates(
        classes=classes, features=features, sample=0.05, shuffle=True
    )

    # Fit and transform the data to the visualizer
    visualizer.fit_transform(X, y)

    # Finalize the title and axes then display the visualization
    visualizer.finalize()
    visualizer.ax.grid(False)


By inspecting the visualization closely, we can see that the combination of transparency and overlap gives us the sense of groups of similar instances, sometimes referred to as "braids". If there are distinct braids of different classes, it suggests that there is enough separability that a classification algorithm might be able to discern between each class.

Unfortunately, as we inspect this class, we can see that the domain of each feature may make the visualization hard to interpret. In the above visualization, the domain of the ``light`` feature is from in ``[0, 1600]``, far larger than the range of temperature in ``[50, 96]``. To solve this problem, each feature should be scaled or normalized so they are approximately in the same domain.

Normalization techniques can be directly applied to the visualizer without pre-transforming the data (though you could also do this) by using the ``normalize`` parameter. Several transformers are available; try using ``minmax``, ``minabs``, ``standard``, ``l1``, or ``l2`` normalization to change perspectives in the parallel coordinates as follows:

.. plot::

    from yellowbrick.features import ParallelCoordinates
    from yellowbrick.datasets import load_occupancy

    # Load the classification data set
    X, y = load_occupancy()

    # Specify the features of interest and the classes of the target
    features = [
        "temperature", "relative humidity", "light", "C02", "humidity"
    ]
    classes = ["unoccupied", "occupied"]

    # Instantiate the visualizer
    visualizer = ParallelCoordinates(
        classes=classes, features=features,
        normalize='standard', sample=0.05, shuffle=True,
    )

    # Fit the visualizer and display it
    visualizer.fit_transform(X, y)
    visualizer.finalize()
    visualizer.ax.grid(False)


Now we can see that each feature is in the range ``[-3, 3]`` where the mean of the feature is set to zero and each feature has a unit variance applied between ``[-1, 1]`` (because we're using the ``StandardScaler`` via the ``standard`` normalize parameter). This version of parallel coordinates gives us a much better sense of the distribution of the features and if any features are highly variable with respect to any one class.


Faster Parallel Coordinates
---------------------------

Parallel coordinates can take a long time to draw since each instance is represented by a line for each feature. Worse, this time is not well spent since a lot of overlap in the visualization makes the parallel coordinates less understandable. We propose two solutions to this:

1. Use ``sample=0.2`` and ``shuffle=True`` parameters to shuffle and sample the dataset being drawn on the figure. The sample parameter will perform a uniform random sample of the data, selecting the percent specified.

2. Use the ``fast=True`` parameter to enable "fast drawing mode".

The "fast" drawing mode vastly improves the performance of the parallel coordinates drawing algorithm by drawing each line segment by class rather than each instance individually. However, this improved performance comes at a cost, as the visualization produced is subtly different; compare the visualizations in fast and standard drawing modes below:

.. plot::
    :include-source: False
    
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from yellowbrick.features import ParallelCoordinates

    data = load_iris()

    _, axes = plt.subplots(nrows=2, figsize=(9,9))

    for idx, fast in enumerate((False, True)):
        title = "Fast Parallel Coordinates" if fast else "Standard Parallel Coordinates"
        oz = ParallelCoordinates(ax=axes[idx], fast=fast, title=title)
        oz.fit_transform(data.data, data.target)
        oz.finalize()
        oz.ax.grid(False)

    plt.tight_layout()

As you can see the "fast" drawing algorithm does not have the same build up of color density where instances of the same class intersect. Because there is only one line per class, there is only a darkening effect between classes. This can lead to a different interpretation of the plot, though it still may be effective for analytical purposes, particularly when you're plotting a lot of data. Needless to say, the performance benefits are dramatic:

.. plot::
    :include-source: False

    import time
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from yellowbrick.features import ParallelCoordinates
    import pandas as pd

    trials = 5
    factors = np.arange(1, 11)

    def pcoords_time(X, y, fast=True):
        _, ax = plt.subplots()
        oz = ParallelCoordinates(fast=fast, ax=ax)

        start = time.time()
        oz.fit_transform(X, y)
        delta = time.time() - start

        plt.cla()        # clear current axis
        plt.clf()        # clear current figure
        plt.close("all") # close all existing plots

        return delta

    def pcoords_speedup(X, y):
        fast_time = pcoords_time(X, y, fast=True)
        slow_time = pcoords_time(X, y, fast=False)

        return slow_time / fast_time

    data = load_iris()

    speedups = []
    variance = []

    for factor in factors:
        X = np.repeat(data.data, factor, axis=0)
        y = np.repeat(data.target, factor, axis=0)

        local_speedups = []
        for trial in range(trials):
            local_speedups.append(pcoords_speedup(X, y))

        local_speedups = np.array(local_speedups)
        speedups.append(local_speedups.mean())
        variance.append(local_speedups.std())

    speedups = np.array(speedups)
    variance = np.array(variance)

    series = pd.Series(speedups, index=factors)
    _, ax = plt.subplots(figsize=(9,6))
    series.plot(ax=ax, marker='o', label="speedup factor", color='b')

    # Plot one standard deviation above and below the mean
    ax.fill_between(
        factors, speedups - variance, speedups + variance, alpha=0.25,
        color='b',
    )

    ax.set_ylabel("speedup factor")
    ax.set_xlabel("dataset size (number of repeats in Iris dataset)")
    ax.set_title("Speed Improvement of Fast Parallel Coordinates")
    ax.grid(True)
    ax.margins(0)



API Reference
-------------

.. automodule:: yellowbrick.features.pcoords
    :members: ParallelCoordinates
    :undoc-members:
    :show-inheritance:
