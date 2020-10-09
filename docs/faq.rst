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
-------------------------------------------------

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

Yellowbrick uses colors to make visualzers as interpretable as possible for intuitive machine learning diagnostics. Generally, color is specified by the target variable, ``y`` that you might pass to an estimator's fit method. Therefore Yellowbrick considers color based on the datatype of the target:

- **Discrete**: when the target is represented by discrete classes, Yellowbrick uses categorical colors that are easy to discriminate from each other.
- **Continuous**: when the target is represented by continous values, Yellowbrick uses a sequential colormap to show the range of data.

*Most* visualizers therefore accept the ``colors`` and ``colormap`` arguments when they are initialized. Generally speaking, if the target is discrete, specify `colors` as a list of valid matplotlib colors; otherwise if your target is continuous, specify a matplotlib colormap or colormap name. Most Yellowbrick visualizers are smart enough to figure out the colors for each of your data points based on what you pass in; for example if you pass in a colormap for a discrete target, the visualizer will create a list of discrete colors from the sequential colors.

.. note:: Although most visualizers support these arguments, please be sure to check the visualizer as it may have specific color requirements. E.g. the :doc:`ResidualsPlot <api/regressor/residuals>` accepts the ``train_color``, ``test_color``, and ``line_color`` to modify its visualization. To see a visualizer's arguments you can use ``help(Visualizer)`` or ``visualizer.get_params()``.


For example, the :doc:`Manifold <api/features/manifold>` can visualize both discrete and sequential targets. In the discrete case, pass a list of `valid color values <https://matplotlib.org/api/colors_api.html>`_ as follows:


.. code:: python

    from yellowbrick.features.manifold import Manifold

    visualizer = Manifold(
        manifold="tsne", target="discrete", colors=["teal", "orchid"]
    )

    ...


... whereas for ``continuous`` targets, it is better to specify a `matplotlib colormap <https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html>`_:


.. code:: python

    from yellowbrick.features.manifold import Manifold

    visualizer = Manifold(
        manifold="isomap", target="continuous", colormap="YlOrRd"
    )

    ...


Finally please note that you can manipulate the default colors that Yellowbrick uses by modifying the `matplotlib styles <https://matplotlib.org/3.1.1/tutorials/introductory/customizing.html>`_, particularly the default color cycle. Yellowbrick also has some tools for style management, please see :doc:`api/palettes` for more information.


How can I save a Yellowbrick plot?
----------------------------------

Save your Yellowbrick plot by passing an ``outpath`` into ``show()``:

.. code:: python

    from sklearn.cluster import MiniBatchKMeans
    from yellowbrick.cluster import KElbowVisualizer

    visualizer = KElbowVisualizer(MiniBatchKMeans(), k=(4,12))

    visualizer.fit(X)
    visualizer.show(outpath="kelbow_minibatchkmeans.png")

    ...

Most backends support png, pdf, ps, eps and svg to save your work!


How can I make overlapping points show up better?
-------------------------------------------------

You can use the ``alpha`` param to change the opacity of plotted points (where ``alpha=1`` is complete opacity, and ``alpha=0`` is complete transparency):

.. code:: python

    from yellowbrick.contrib.scatter import ScatterVisualizer

    visualizer = ScatterVisualizer(
        x="light", y="C02", classes=classes, alpha=0.5
    )


How can I access the sample datasets used in the examples?
----------------------------------------------------------

Visit the :doc:`api/datasets/index` page.


Can I use Yellowbrick with libraries other than scikit-learn?
-------------------------------------------------------------

Potentially! Yellowbrick visualizers rely on the internal model implementing the scikit-learn API (e.g. having a ``fit()`` and ``predict()`` method), and often expect to be able to retrieve learned attributes from the model (e.g. ``coef_``). Some third-party estimators fully implement the scikit-learn API, but not all do.

When using third-party libraries with Yellowbrick, we encourage you to ``wrap`` the model using the ``yellowbrick.contrib.wrapper`` module. Visit the :doc:`api/contrib/wrapper` page for all the details!
