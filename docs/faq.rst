.. -*- mode: rst -*-

Frequently Asked Questions
==========================

Welcome to our frequently asked questions page. We're glad that you're using Yellowbrick! If your question is not captured here, please submit it to our `Google Groups Listserv <https://groups.google.com/forum/#!forum/yellowbrick>`_. This is an email list/forum that you, as a Yellowbrick user, can join and interact with other users to address and troubleshoot Yellowbrick issues. The Google Groups Listserv is where you should be able to receive the quickest response. We would welcome and encourage you to join the group so that you can respond to others' questions! You can also ask questions on `Stack Overflow <http://stackoverflow.com/questions/tagged/yellowbrick>`_ and tag them with "yellowbrick". Finally, you can add issues on GitHub and you can tweet or direct message us on Twitter `@scikit_yb <https://twitter.com/scikit_yb>`_.


How can I change the size of a Yellowbrick plot?
------------------------------------------------

You can change the ``size`` of a plot by passing in the desired dimensions in pixels on instantiation of the visualizer:

.. code:: python

    # Import the visualizer
    from yellowbrick.features import RadViz

    # Instantiate the visualizer using the ``size`` param
    visualizer = RadViz(
        classes=classes, features=features, size=(1080, 720)
    )

    ...


Note: we are considering adding support for passing in ``size`` in inches in a future Yellowbrick release. For a convenient inch-to-pixel converter, check out `www.unitconversion.org <http://www.unitconversion.org/typography/inchs-to-pixels-y-conversion.html>`_.

How can I change the title of a Yellowbrick plot?
---------------------------------------------------

You can change the ``title`` of a plot by passing in the desired title as a string on instantiation:


.. code:: python

    from yellowbrick.classifier import ROCAUC
    from sklearn.linear_model import RidgeClassifier

    # Create your custom title
    my_title = "ROCAUC Curves for Multiclass RidgeClassifier"

    # Instantiate the visualizer passing the custom title
    visualizer = ROCAUC(
        RidgeClassifier(), classes=classes, title=my_title
    )

    ...


How can I change the color of a Yellowbrick plot?
-------------------------------------------------

To customize coloring in your plot, use the ``colors`` or ``cmap`` (or ``colormap``) arguments. Note that different visualizers may require slightly different arguments depending on how they construct the plots.

For instance, the :doc:`api/features/manifold` accepts a ``colors`` argument, for which ``discrete`` targets should be the name of one of the :doc:`api/palettes` or a list of `matplotlib colors <https://matplotlib.org/examples/color/named_colors.html>`_ represented as strings:
For instance, the :doc:`api/features/manifold` accepts a ``colors`` argument, for which ``discrete`` targets should be the name of a palette from the Yellowbrick :doc:`api/palettes` or a list of `matplotlib colors <https://matplotlib.org/examples/color/named_colors.html>`_ represented as strings:

.. code:: python

    from yellowbrick.features.manifold import Manifold

    visualizer = Manifold(
        manifold="tsne", target="discrete", colors=["teal", "orchid"]
    )

    ...


... whereas for ``continuous`` targets, ``colors`` should be a colormap:


.. code:: python

    from yellowbrick.features.manifold import Manifold

    visualizer = Manifold(
        manifold="isomap", target="continuous", colors="YlOrRd"
    )

    ...


Other visualizers accept a ``cmap`` argument:

.. code:: python

    from sklearn.linear_model import LogisticRegression
    from yellowbrick.classifier import ConfusionMatrix

    visualizer = ConfusionMatrix(
        LogisticRegression(), cmap="YlGnBu"
    )

    ...

Or a ``colormap`` argument:

.. code:: python

    from yellowbrick.features import ParallelCoordinates

    # Instantiate the visualizer
    visualizer = ParallelCoordinates(
        classes=classes, features=features, colormap="PRGn"
    )

    ...

The :doc:`api/regressor/residuals` accepts color argument for the training and test points, ``train_color`` and ``test_color``, respectively:

.. code:: python

    from yellowbrick.regressor import ResidualsPlot
    from sklearn.linear_model import ElasticNet

    visualizer = ResidualsPlot(
        model=ElasticNet()
        train_color=train_color,  # color of points model was trained on
        test_color=train_color,   # color of points model was tested on
        line_color=line_color    # color of zero-error line
    )


How can I save a Yellowbrick plot?
-----------------------------------

Save your Yellowbrick plot by passing an ``outpath`` into ``poof()``:

.. code:: python

    from sklearn.cluster import MiniBatchKMeans
    from yellowbrick.cluster import KElbowVisualizer

    visualizer = KElbowVisualizer(MiniBatchKMeans(), k=(4,12))

    visualizer.fit(X)
    visualizer.poof(outpath="kelbow_minibatchkmeans.png")

    ...

Most backends support png, pdf, ps, eps and svg to save your work!


How can I make overlapping points show up better?
----------------------------------------------------

You can use the ``alpha`` param to change the opacity of plotted points (where ``alpha=1`` is complete opacity, and ``alpha=0`` is complete transparency):

.. code:: python

    from yellowbrick.contrib.scatter import ScatterVisualizer

    visualizer = ScatterVisualizer(
        x="light", y="C02", classes=classes, alpha=0.5
    )


How can I access the sample datasets used in the examples?
---------------------------------------------------------------

Visit the :doc:`api/datasets` page.
