.. -*- mode: rst -*-

RadViz Visualizer
=================

``RadViz`` is a multivariate data visualization algorithm that plots each
feature dimension uniformly around the circumference of a circle then
plots points on the interior of the circle such that the point
normalizes its values on the axes from the center to each arc. This
mechanism allows as many dimensions as will easily fit on a circle,
greatly expanding the dimensionality of the visualization.

Data scientists use this method to detect separability between classes.
E.g. is there an opportunity to learn from the feature set or is there
just too much noise?

If your data contains rows with missing values (``numpy.nan``), those missing
values will not be plotted. In other words, you may not get the entire
picture of your data. ``RadViz`` will raise a ``DataWarning`` to inform you of the
percent missing.

If you do receive this warning, you may want to look at imputation strategies.
A good starting place is the `scikit-learn Imputer. <http://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html>`_

=================   =================
Visualizer           :class:`~yellowbrick.features.radviz.RadialVisualizer`
Quick Method         :func:`~yellowbrick.features.radviz.radviz`
Models               Classification, Regression
Workflow             Feature Engineering
=================   =================

.. plot::
    :context: close-figs
    :alt: RadViz on the Occupancy Dataset

    from yellowbrick.datasets import load_occupancy
    from yellowbrick.features import RadViz

    # Load the classification dataset
    X, y = load_occupancy()

    # Specify the target classes
    classes = ["unoccupied", "occupied"]

    # Instantiate the visualizer
    visualizer = RadViz(classes=classes)

    visualizer.fit(X, y)           # Fit the data to the visualizer
    visualizer.transform(X)        # Transform the data
    visualizer.show()              # Finalize and render the figure

For regression, the ``RadViz`` visualizer should use a color sequence to
display the target information, as opposed to discrete colors.


Quick Method
------------

The same functionality above can be achieved with the associated quick method ``radviz``. This method will build the ``RadViz`` object with the associated arguments, fit it, then (optionally) immediately show the visualization.

.. plot::
    :context: close-figs
    :alt: radviz on the occupancy dataset

    from yellowbrick.features.radviz import radviz
    from yellowbrick.datasets import load_occupancy

    #Load the classification dataset
    X, y = load_occupancy()

    # Specify the target classes
    classes = ["unoccupied", "occupied"]

    # Instantiate the visualizer
    radviz(X, y, classes=classes)


API Reference
-------------

.. automodule:: yellowbrick.features.radviz
    :members: RadialVisualizer, RadViz, radviz
    :undoc-members:
    :show-inheritance:
