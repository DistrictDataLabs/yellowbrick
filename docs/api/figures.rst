.. -*- mode: rst -*-

Figures and Axes
================

This document is an open letter to the PyData community, particularly those that are involved in matplotlib development. We'd like to get some advice on the API choice we've made and thoughts about our use of the matplotlib Axes objects.

One of the most complex parts of designing a visualization library around matplotlib is working with figures and axes. As defined in `The Lifecycle of a Plot <https://matplotlib.org/tutorials/introductory/lifecycle.html>`_, these central objects of matplotlib plots are as follows:

- A Figure is the final image that may contain 1 or more Axes.
- An Axes represents an individual plot

Based on these definitions and and the advice to "try to use the object-oriented interface over the pyplot interface", the Yellowbrick interface is designed to wrap a matplotlib ``axes.Axes``. We propose the following general use case for most visualizers:

.. code:: python

    import matplotlib.pyplot as plt
    from yellowbrick import Visualizer, quick_visualizer

    fig, ax = plt.subplots()

    # Object oriented approach
    viz = Visualizer(ax=ax)
    viz.fit(X, y)
    viz.show()

    # Quick method approach
    viz = quick_visualizer(X, y, ax=ax)
    viz.show()

This design allows users to more directly control the size, style, and interaction with the plot (though YB does provide some helpers for these as well). For example, if a user wanted to generate a report with multiple visualizers for a classification problem, it may looks something like:

.. code:: python

    import matplotlib.pyplot as plt

    from yellowbrick.features import FeatureImportances
    from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC
    from sklearn.linear_model import LogisticRegression

    fig, axes = plot.subplots(2, 2)

    model = LogisticRegression()
    visualgrid = [
        FeatureImportances(ax=axes[0][0]),
        ConfusionMatrix(model, ax=axes[0][1]),
        ClassificationReport(model, ax=axes[1][0]),
        ROCAUC(model, ax=axes[1][1]),
    ]

    for viz in visualgrid:
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()

    plt.show()

This is a common use case and we're working on the idea of "visual pipelines" to support this type of development because, for machine learning, users generally want a suite of visualizers or a report, not just a single visualization. The API requirement to support this has therefore been that visualizers use the ``ax`` object passed to them and not ``plt``. If the user does not pass a specific ``ax`` then the global current axes is used via ``plt.gca``. Generally, visualizers should behave as though they are a plot that as part of a larger figure.

Visualizers are getting more complex, however, and some are becoming multi-axes plots in their own right. For example:

- The ResidualsPlot has a scatter plot axes and a histogram axes
- The JointPlot has a scatter plot and two histogram axes
- Data driven scatter plot axes often have colorbar axes
- The PCA plot has scatter plot, color bar, and heatmap axes
- The confusion matrix probability histogram is a grid of axes for each class pair
- The ICDM has an inset axes that acts as a dynamic legend

Although it would have been easier to simply embed the figure into the visualizer and use a ``GridSpec`` or other layout tool, the focus on ensuring visualizers are individual plots that wrap an Axes has made us bend over backward to adjust the plot inside of the axes area that was originally supplied, primarily by using ``make_axes_locateable``, which is part of the AxesGrid toolkit.

Generally, it appears that the `AxesGrid Toolkit <https://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#imagegrid>`_ is the right tool for Yellowbrick - many of the examples shown are similar to the things that Yellowbrick is trying to do. However, this package is not fully documented with examples and some helper utilities that would be useful, for example the ``ImageGrid``, still require a ``figure.Figure`` object.

At this point we are left with some important questions about Yellowbrick's development roadmap:

1. Like Seaborn, should YB have two classes of visualizer, one that wraps an axes and one that wraps a figure?
2. Should we go all in on the AxesGrid toolkit and continue to restrict our use of the figure, will this method be supported in the long run?


Other notes and discussion:

- `Create equal aspect (square) plot with multiple axes when data limits are different? <https://stackoverflow.com/questions/54545758/create-equal-aspect-square-plot-with-multiple-axes-when-data-limits-are-differ>`_
