.. -*- mode: rst -*-

Developing Visualizers
======================

In this section, we'll discuss the basics of developing visualizers. This of course is a big topic, but hopefully these simple tips and tricks will help make sense. First thing though, check out this presentation that we put together on yellowbrick development, it discusses the expected user workflow, our integration with scikit-learn, our plans and roadmap, etc:

.. raw:: html

    <iframe src="https://www.slideshare.net/BenjaminBengfort/slideshelf" width="615px" height="470px" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:none;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe>

One thing that is necessary is a good understanding of scikit-learn and Matplotlib. Because our API is intended to integrate with scikit-learn, a good start is to review `"APIs of scikit-learn objects" <http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects>`_ and `"rolling your own estimator" <http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator>`_. In terms of matplotlib, use Yellowbrick's guide :doc:`../matplotlib`. Additional resources include `Nicolas P. Rougier's Matplotlib tutorial <https://www.labri.fr/perso/nrougier/teaching/matplotlib/>`_ and `Chris Moffitt's Effectively Using Matplotlib <http://pbpython.com/effective-matplotlib.html>`_.

Visualizer API
--------------

There are two basic types of Visualizers:

- **Feature Visualizers** are high dimensional data visualizations that are essentially transformers.
- **Score Visualizers** wrap a scikit-learn regressor, classifier, or clusterer and visualize the behavior or performance of the model on test data.

These two basic types of visualizers map well to the two basic objects in scikit-learn:

- **Transformers** take input data and return a new data set.
- **Estimators** are fit to training data and can make predictions.

The scikit-learn API is object oriented, and estimators and transformers are initialized with parameters by instantiating their class. Hyperparameters can also be set using the ``set_attrs()`` method and retrieved with the corresponding ``get_attrs()`` method. All scikit-learn estimators have a ``fit(X, y=None)`` method that accepts a two dimensional data array, ``X``, and optionally a vector ``y`` of target values. The ``fit()`` method trains the estimator, making it ready to transform data or make predictions. Transformers have an associated ``transform(X)`` method that returns a new dataset, ``Xprime`` and models have a ``predict(X)`` method that returns a vector of predictions, ``yhat``. Models also have a ``score(X, y)`` method that evaluate the performance of the model.

Visualizers interact with scikit-learn objects by intersecting with them at the methods defined above. Specifically, visualizers perform actions related to ``fit()``, ``transform()``, ``predict()``, and ``score()`` then call a ``draw()`` method which initializes the underlying figure associated with the visualizer. The user calls the visualizer's ``poof()`` method, which in turn calls a ``finalize()`` method on the visualizer to draw legends, titles, etc. and then ``poof()`` renders the figure. The Visualizer API is therefore:

- ``draw()``: add visual elements to the underlying axes object
- ``finalize()``: prepare the figure for rendering, adding final touches such as legends, titles, axis labels, etc.
- ``poof()``: render the figure for the user (or saves it to disk).

Creating a visualizer means defining a class that extends ``Visualizer`` or one of its subclasses, then implementing several of the methods described above. A barebones implementation is as follows::

    import matplotlib.pyplot as plt

    from yellowbrick.base import Visualizer

    class MyVisualizer(Visualizer):

        def __init__(self, ax=None, **kwargs):
            super(MyVisualizer, self).__init__(ax, **kwargs)

        def fit(self, X, y=None):
            self.draw(X)
            return self

        def draw(self, X):
            if self.ax is None:
                self.ax = self.gca()

            self.ax.plt(X)

        def finalize(self):
            self.set_title("My Visualizer")

This simple visualizer simply draws a line graph for some input dataset X, intersecting with the scikit-learn API at the ``fit()`` method. A user would use this visualizer in the typical style::

    visualizer = MyVisualizer()
    visualizer.fit(X)
    visualizer.poof()

Score visualizers work on the same principle but accept an additional required ``model`` argument. Score visualizers wrap the model (which can be either instantiated or uninstantiated) and then pass through all attributes and methods through to the underlying model, drawing where necessary.

Testing
-------

The test package mirrors the yellowbrick package in structure and also contains several helper methods and base functionality. To add a test to your visualizer, find the corresponding file to add the test case, or create a new test file in the same place you added your code.

Visual tests are notoriously difficult to create --- how do you test a visualization or figure? Moreover, testing scikit-learn models with real data can consume a lot of memory. Therefore the primary test you should create is simply to test your visualizer from end to end and make sure that no exceptions occur. To assist with this, we have two primary helpers, ``VisualTestCase`` and ``DatasetMixin``. Create your unittest as follows::

    import pytest
    from tests.base import VisualTestCase
    from tests.dataset import DatasetMixin

    class MyVisualizerTests(VisualTestCase, DatasetMixin):

        def test_my_visualizer(self):
            """
            Test MyVisualizer on a real dataset
            """
            # Load the data from the fixture
            dataset = self.load_data('occupancy')

            # Get the data
            X = dataset[[
                "temperature", "relative_humidity", "light", "C02", "humidity"
            ]]
            y = dataset['occupancy'].astype(int)

            try:
                visualizer = MyVisualizer()
                visualizer.fit(X)
                visualizer.poof()
            except Exception as e:
                pytest.fail("my visualizer didn't work")

Tests can be run as follows::

    $ make test

The Makefile uses the pytest runner and testing suite as well as the coverage library, so make sure you have those dependencies installed! The ``DatasetMixin`` also requires `requests.py <http://docs.python-requests.org/en/master/>`_ to fetch data from our Amazon S3 account.

Image Comparison Tests
----------------------

Writing an image based comparison test is only a little more difficult than the simple testcase presented above. We have adapted matplotlib's image comparison test utility into an easy to use assert method : ``self.assert_images_similar(visualizer)``

The main consideration is that you must specify the “baseline”, or expected, image in the ``tests/baseline_images/`` folder structure.

For example, create your unittest located in ``tests/test_regressor/test_myvisualizer.py`` as follows::

    from tests.base import VisualTestCase
    ...
        def test_my_visualizer_output(self):
            ...
            visualizer = MyVisualizer()
            visualizer.fit(X)
            visualizer.poof()
            self.assert_images_similar(visualizer)

The first time this test is run, there will be no baseline image to compare against, so the test will fail. Copy the output images (in this case ``tests/actual_images/test_regressor/test_myvisualizer/test_my_visualizer_output.png``) to the correct subdirectory of baseline_images tree in the source directory (in this case ``tests/baseline_images/test_regressor/test_myvisualizer/test_my_visualizer_output.png``). Put this new file under source code revision control (with git add). When rerunning the tests, they should now pass.

We also have a helper script, ``tests/images.py`` to clean up and manage baseline images automatically. It is run using the ``python -m`` command to execute a module as main, and it takes as an argument the path to your *test file*. To copy the figures as above::

    $ python -m tests.images tests/test_regressor/test_myvisualizer.py

This will move all related test images from ``actual_images`` to ``baseline_images`` on your behalf (note you'll have had to run the tests at least once to generate the images). You can also clean up images from both actual and baseline as follows::

    $ python -m tests.images -C tests/test_regressor/test_myvisualizer.py

This is useful particularly if you're stuck trying to get an image comparison to work. For more information on the images helper script, use ``python -m tests.images --help``.

.. _documentation:

Documentation
-------------

The initial documentation for your visualizer will be a well structured docstring. Yellowbrick uses Sphinx to build documentation, therefore docstrings should be written in reStructuredText in numpydoc format (similar to scikit-learn). The primary location of your docstring should be right under the class definition, here is an example::

    class MyVisualizer(Visualizer):
        """
        This initial section should describe the visualizer and what
        it's about, including how to use it. Take as many paragraphs
        as needed to get as much detail as possible.

        In the next section describe the parameters to __init__.

        Parameters
        ----------

        model : a scikit-learn regressor
            Should be an instance of a regressor, and specifically one whose name
            ends with "CV" otherwise a will raise a YellowbrickTypeError exception
            on instantiation. To use non-CV regressors see:
            ``ManualAlphaSelection``.

        ax : matplotlib Axes, default: None
            The axes to plot the figure on. If None is passed in the current axes
            will be used (or generated if required).

        kwargs : dict
            Keyword arguments that are passed to the base class and may influence
            the visualization as defined in other Visualizers.

        Examples
        --------

        >>> model = MyVisualizer()
        >>> model.fit(X)
        >>> model.poof()

        Notes
        -----

        In the notes section specify any gotchas or other info.
        """

When your visualizer is added to the API section of the documentation, this docstring will be rendered in HTML to show the various options and functionality of your visualizer!

To add the visualizer to the documentation it needs to be added to the ``docs/api`` folder in the correct subdirectory. For example if your visualizer is a model score visualizer related to regression it would go in the ``docs/api/regressor`` subdirectory. If you have a question where your documentation should be located, please ask the maintainers via your pull request, we'd be happy to help!

There are two primary files that need to be created:

1. **mymodule.rst**: the reStructuredText document
2. **mymodule.py**: a python file that generates images for the rst document

There are quite a few examples in the documentation on which you can base your files of similar types. The primary format for the API section is as follows::

    .. -*- mode: rst -*-

    My Visualizer
    =============

    Intro to my visualizer

    .. code:: python

        # Example to run MyVisualizer
        visualizer = MyVisualizer(LinearRegression())

        visualizer.fit(X, y)
        g = visualizer.poof()


    .. image:: images/my_visualizer.png

    Discussion about my visualizer


    API Reference
    -------------

    .. automodule:: yellowbrick.regressor.mymodule
        :members: MyVisualizer
        :undoc-members:
        :show-inheritance:

This is a pretty good structure for a documentation page; a brief introduction followed by a code example with a visualization included (using the ``mymodule.py`` to generate the images into the local directory's ``images`` subdirectory). The primary section is wrapped up with a discussion about how to interpret the visualizer and use it in practice. Finally the ``API Reference`` section will use ``automodule`` to include the documentation from your docstring.

At this point there are several places where you can list your visualizer, but to ensure it is included in the documentation it *must be listed in the TOC of the local index*. Find the ``index.rst`` file in your subdirectory and add your rst file (without the ``.rst`` extension) to the ``..toctree::`` directive. This will ensure the documentation is included when it is built.

Speaking of, you can build your documentation by changing into the ``docs`` directory and running ``make html``, the documentation will be built and rendered in the ``_build/html`` directory. You can view it by opening ``_build/html/index.html`` then navigating to your documentation in the browser.

There are several other places that you can list your visualizer including:

 - ``docs/index.rst`` for a high level overview of our visualizers
 - ``DESCRIPTION.rst`` for inclusion on PyPI
 - ``README.md`` for inclusion on GitHub

Please ask for the maintainer's advice about how to include your visualizer in these pages.
