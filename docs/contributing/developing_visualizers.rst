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

These two basic types of visualizers map well to the two basic estimator objects in scikit-learn:

- **Transformers** take input data and return a new data set.
- **Models** are fit to training data and can make predictions.

The scikit-learn API is object oriented, and estimators are initialized with parameters by instantiating their class. Hyperparameters can also be set using the ``set_attrs()`` method and retrieved with the corresponding ``get_attrs()`` method. All scikit-learn estimators have a ``fit(X, y=None)`` method that accepts a two dimensional data array, ``X``, and optionally a vector ``y`` of target values. The ``fit()`` method trains the estimator, making it ready to transform data or make predictions. Transformers have an associated ``transform(X)`` method that returns a new dataset, ``Xprime`` and models have a ``predict(X)`` method that returns a vector of predictions, ``yhat``. Models may also have a ``score(X, y)`` method that evaluate the performance of the model.

Visualizers interact with scikit-learn objects by intersecting with them at the methods defined above. Specifically, visualizers perform actions related to ``fit()``, ``transform()``, ``predict()``, and ``score()`` then call a ``draw()`` method which initializes the underlying figure associated with the visualizer. The user calls the visualizer's ``show()`` method, which in turn calls a ``finalize()`` method on the visualizer to draw legends, titles, etc. and then ``show()`` renders the figure. The Visualizer API is therefore:

- ``draw()``: add visual elements to the underlying axes object
- ``finalize()``: prepare the figure for rendering, adding final touches such as legends, titles, axis labels, etc.
- ``show()``: render the figure for the user (or saves it to disk).

Creating a visualizer means defining a class that extends ``Visualizer`` or one of its subclasses, then implementing several of the methods described above. A barebones implementation is as follows:

.. code:: python

    import matplotlib.pyplot as plt

    from yellowbrick.base import Visualizer

    class MyVisualizer(Visualizer):

        def __init__(self, ax=None, **kwargs):
            super(MyVisualizer, self).__init__(ax, **kwargs)

        def fit(self, X, y=None):
            self.draw(X)
            return self

        def draw(self, X):
            self.ax.plt(X)
            return self.ax

        def finalize(self):
            self.set_title("My Visualizer")

This simple visualizer simply draws a line graph for some input dataset X, intersecting with the scikit-learn API at the ``fit()`` method. A user would use this visualizer in the typical style:

.. code:: python

    visualizer = MyVisualizer()
    visualizer.fit(X)
    visualizer.show()

Score visualizers work on the same principle but accept an additional required ``estimator`` argument. Score visualizers wrap the model (which can be either fitted or unfitted) and then pass through all attributes and methods through to the underlying model, drawing where necessary.

.. code:: python

    from yellowbrick.base import ScoreVisualizer

    class MyScoreVisualizer(ScoreVisualizer):

        def __init__(self, estimator, ax=None, **kwargs):
            super(MyScoreVisualizer, self).__init__(estimator, ax=ax, **kwargs)

        def fit(self, X_train, y_train=None):
            # Fit the underlying model
            super(MyScoreVisualizer, self).fit(X_train, y_train)
            self.draw(X_train, y_train)
            return self

        def score(self, X_test, y_test):
            # Score the underlying model
            super(MyScoreVisualizer, self).fit(X_train, y_train)
            self.draw(X_test, y_test)
            return self.score_

        def draw(self, X, y):
            self.ax.scatter(X, c=y)
            return self.ax

        def finalize(self):
            self.set_title("My Score Visualizer")

Note that the calls to ``super`` in the above code ensure that the base functionality (e.g. fitting a model and computing the score) are required to ensure the visualizer is consistent with other visualizers.

Datasets
--------

Yellowbrick gives easy access to several datasets that are used for the examples in the documentation and testing. These datasets are hosted in our CDN and must be downloaded for use. Typically, when a user calls one of the data loader functions, e.g. ``load_bikeshare()`` the data is automatically downloaded if it's not already on the user's computer. However, for development and testing, or if you know you will be working without internet access, it might be easier to simply download all the data at once.

The data downloader script can be run as follows::

    $ python -m yellowbrick.download

This will download the data to the fixtures directory inside of the Yellowbrick site packages. You can specify the location of the download either as an argument to the downloader script (use ``--help`` for more details) or by setting the ``$YELLOWBRICK_DATA`` environment variable. This is the preferred mechanism because this will also influence how data is loaded in Yellowbrick.

Note that developers who have downloaded data from Yellowbrick versions earlier than v1.0 may experience some problems with the older data format. If this occurs, you can clear out your data cache as follows::

    $ python -m yellowbrick.download --cleanup

This will remove old datasets and download the new ones. You can also use the ``--no-download`` flag to simply clear the cache without re-downloading data. Users who are having difficulty with datasets can also use this or they can uninstall and reinstall Yellowbrick using ``pip``.

Testing
-------

The test package mirrors the yellowbrick package in structure and also contains several helper methods and base functionality. To add a test to your visualizer, find the corresponding file to add the test case, or create a new test file in the same place you added your code.

Visual tests are notoriously difficult to create --- how do you test a visualization or figure? Moreover, testing scikit-learn models with real data can consume a lot of memory. Therefore the primary test you should create is simply to test your visualizer from end to end and make sure that no exceptions occur. To assist with this, we have a helper, ``VisualTestCase``. Create your tests as follows:

.. code:: python

    import pytest

    from tests.base import VisualTestCase
    from yellowbrick.datasets import load_occupancy

    class MyVisualizerTests(VisualTestCase):

        def test_my_visualizer(self):
            """
            Test MyVisualizer on a real dataset
            """
            # Load the occupancy dataset
            X, y = load_occupancy()

            try:
                visualizer = MyVisualizer()
                assert visualizer.fit(X, y) is visualizer, "fit should return self"
                visualizer.show()
            except Exception as e:
                pytest.fail("my visualizer didn't work: {}".format(e))

This simple test case is an excellent start to a larger test package and we recommend starting with this test as you develop your visualizer. Once you've completed the development and prototyping you can start to include :ref:`test fixtures <fixtures>` and test various normal use cases and edge cases with unit tests, then build :ref:`image similarity tests <assert_images_similar>` to more thoroughly define the integration tests.


Running the Test Suite
~~~~~~~~~~~~~~~~~~~~~~

To run the test suite, first install the testing dependencies that are located in the `tests` folder as follows::

    $ pip install -r tests/requirements.txt

The required dependencies for the test suite include testing utilities and libraries such as pandas and nltk that are not included in the core dependencies.

Tests can be run as follows from the project root::

    $ make test

The Makefile uses the pytest runner and testing suite as well as the coverage library.

.. _assert_images_similar:

Image Comparison Tests
~~~~~~~~~~~~~~~~~~~~~~

Writing an image based comparison test is only a little more difficult than the simple testcase presented above. We have adapted matplotlib's image comparison test utility into an easy to use assert method : ``self.assert_images_similar(visualizer)``

The main consideration is that you must specify the “baseline”, or expected, image in the ``tests/baseline_images/`` folder structure.

For example, create your test function located in ``tests/test_regressor/test_myvisualizer.py`` as follows:

.. code:: python

    from tests.base import VisualTestCase

    class MyVisualizerTests(VisualTestCase):

        def test_my_visualizer_output(self):
            visualizer = MyVisualizer()
            visualizer.fit(X)
            visualizer.show()
            self.assert_images_similar(visualizer)

The first time this test is run, there will be no baseline image to compare against, so the test will fail. Copy the output images (in this case ``tests/actual_images/test_regressor/test_myvisualizer/test_my_visualizer_output.png``) to the correct subdirectory of baseline_images tree in the source directory (in this case ``tests/baseline_images/test_regressor/test_myvisualizer/test_my_visualizer_output.png``). Put this new file under source code revision control (with git add). When rerunning the tests, they should now pass.

We also have a helper script, ``tests/images.py`` to clean up and manage baseline images automatically. It is run using the ``python -m`` command to execute a module as main, and it takes as an argument the path to your *test file*. To copy the figures as above::

    $ python -m tests.images tests/test_regressor/test_myvisualizer.py

This will move all related test images from ``actual_images`` to ``baseline_images`` on your behalf (note you'll have had to run the tests at least once to generate the images). You can also clean up images from both actual and baseline as follows::

    $ python -m tests.images -C tests/test_regressor/test_myvisualizer.py

This is useful particularly if you're stuck trying to get an image comparison to work. For more information on the images helper script, use ``python -m tests.images --help``.

.. _fixtures:

Test Fixtures
~~~~~~~~~~~~~

Often, you will need a controlled dataset to test your visualizer as specifically as possible. To do this, we recommend that you make use of `pytest fixtures <https://docs.pytest.org/en/latest/fixture.html>`_ and `scikit-learn's generated datasets <https://scikit-learn.org/stable/datasets/index.html#generated-datasets>`_. Together these tools ensure that you have complete control over your test fixtures and can test different user scenarios as precisely as possible. For example, consider the case where we want to test both a binary and a multiclass dataset for a classification score visualizer.

.. code:: python

    from tests.fixtures import Dataset, Split

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split as tts

    @pytest.fixture(scope="class")
    def binary(request):
        """
        Creates a random binary classification dataset fixture
        """
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            n_clusters_per_class=3,
            random_state=2001,
        )

        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)

        dataset = Dataset(Split(X_train, X_test), Split(y_train, y_test))
        request.cls.binary = dataset

In this example, we make use of :func:`sklearn.datasets.make_classification` to randomly generate exactly the dataset that we'd like, in this case a dataset with 2 classes and enough variability so as to be interesting. Because we're using this with a score visualizer, it is helpful to divide this into train and test splits. The ``Dataset`` and ``Split`` objects in ``tests.fixtures`` are namedtuples that allow you to easily access ``X`` and ``y`` properties on the dataset and ``train`` and ``test`` properties on the split. Creating a dataset this way means we can access ``dataset.X.train`` and ``dataset.y.test`` easily in our test functions.

Similarly, we can create a custom multiclass function as well:

.. code:: python

    @pytest.fixture(scope="class")
    def multiclass(request):
        """
        Creates a random multiclass classification dataset fixture
        """
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_informative=8,
            n_redundant=2,
            n_classes=6,
            n_clusters_per_class=3,
            random_state=87,
        )

        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=93)

        dataset = Dataset(Split(X_train, X_test), Split(y_train, y_test))
        request.cls.multiclass = dataset

.. note:: Fixtures that are added to ``conftest.py`` are available to tests in the same directory or a subdirectory as ``conftest.py``. This is special pytest magic since fixtures are identified by strings. Note that the two above example fixtures are in ``tests/test_classifier/conftest.py`` so you can use these exactly in the ``tests/test_classifier`` directory without having to create new fixtures.

To use these fixtures with a ``VisualTestCase`` you must decorate the test class with the fixture. Once done, the fixture will be *generated once per class* and stored in the ``request.cls.<property>`` variable. Here's how to use the above fixtures:

.. code:: python

    @pytest.mark.usefixtures("binary", "multiclass")
    class TestMyScoreVisualizer(VisualTestCase):

        def test_binary(self):
            oz = MyScoreVisualizer()
            assert oz.fit(self.binary.X.train, self.binary.y.train) is oz
            assert 0.0 <= oz.score(self.binary.X.test, self.binary.y.test) <= 1.0
            oz.finalize()

            self.assert_images_similar(oz)

In the above test examples, we showed the use of the yellowbrick dataset loaders, e.g. ``load_occupancy()``. You should feel free to use those datasets and the scikit-learn datasets for tests, particularly for integration tests (described next). The use of the generated datasets and fixtures allows a lot of control over what is being tested and ensures that the tests run as quickly as possible, therefore please use fixtures for the majority of test cases.

Integration Tests
~~~~~~~~~~~~~~~~~

The majority of test cases will use generated test fixtures as described above. But as a visualizer is concluded, it is important to create two "integration tests" that use real-world data in the form of Pandas and numpy arrays from the yellowbrick datasets loaders. These tests often take the following form:

.. code:: python

    try:
        import pandas as pd
    except ImportError:
        pd = None

    class MyVisualizerTests(VisualTestCase):

        @pytest.mark.skipif(pd is None, reason="test requires pandas")
        def test_pandas_integration(self):
            """
            Test with Pandas DataFrame and Series input
            """
            X, y = load_occupancy(return_datset=True).to_pandas()
            oz = MyScoreVisualizer().fit(X, y)
            oz.finalize()
            self.assert_images_similar(oz)

        def test_numpy_integration(self):
            """
            Test with numpy arrays
            """
            X, y = load_occupancy(return_datset=True).to_numpy()
            oz = MyScoreVisualizer().fit(X, y)
            oz.finalize()
            self.assert_images_similar(oz)

These tests often offer the most complications with your visual test cases, so be sure to reserve them for the last tests you create!

.. _documentation:

Documentation
-------------

Yellowbrick uses `Sphinx <http://www.sphinx-doc.org/en/master/index.html>`_ to build our documentation. The advantages of using Sphinx are many; we can more directly link to the documentation and source code of other projects like Matplotlib and scikit-learn using `intersphinx <http://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html>`_. In addition, docstrings used to describe Yellowbrick visualizers can be automatically included when the documentation is built via `autodoc <http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#sphinx.ext.autodoc>`_.

To take advantage of these features, our documentation must be written in reStructuredText (or "rst"). reStructuredText is similar to markdown, but not identical, and does take some getting used to. For instance, styling for things like codeblocks, external hyperlinks, internal cross references, notes, and fixed-width text are all unique in rst.

If you would like to contribute to our documentation and do not have prior experience with rst, we recommend you make use of these resources:

- `A reStructuredText Primer <http://docutils.sourceforge.net/docs/user/rst/quickstart.html>`_
- `rst notes and cheatsheet <https://cheat.readthedocs.io/en/latest/rst.html>`_
- `Using the plot directive <https://matplotlib.org/devel/plot_directive.html>`_

Docstrings
~~~~~~~~~~

The initial documentation for your visualizer will be a well structured docstring. Yellowbrick uses Sphinx to build documentation, therefore docstrings should be written in reStructuredText in numpydoc format (similar to scikit-learn). The primary location of your docstring should be right under the class definition, here is an example:

.. code:: python

    class MyVisualizer(Visualizer):
        """Short description of MyVisualizer

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

        Attributes
        ----------
        score_ : float
            The coefficient of determination that is learned during the visual
            diagnostic, saved for reference after the image has been created.

        Examples
        --------
        >>> model = MyVisualizer()
        >>> model.fit(X)
        >>> model.show()

        Notes
        -----
        In the notes section specify any gotchas or other info.
        """

When your visualizer is added to the API section of the documentation, this docstring will be rendered in HTML to show the various options and functionality of your visualizer!

API Documentation Page
~~~~~~~~~~~~~~~~~~~~~~

To add the visualizer to the documentation it needs to be added to the ``docs/api`` folder in the correct subdirectory. For example if your visualizer is a model score visualizer related to regression it would go in the ``docs/api/regressor`` subdirectory. Add your file named after your module, e.g. ``docs/api/regressor/mymodule.rst``. If you have a question where your documentation should be located, please ask the maintainers via your pull request, we'd be happy to help!

There are quite a few examples in the documentation on which you can base your files of similar types. The primary format for the API section is as follows:

.. code:: rst

    .. -*- mode: rst -*-

    My Visualizer
    =============

    A brief introduction to my visualizer and how it is useful in the machine learning process.

    .. plot::
        :context: close-figs
        :include-source: False
        :alt: Example using MyVisualizer

        visualizer = MyVisualizer(LinearRegression())

        visualizer.fit(X, y)
        g = visualizer.show()

    Discussion about my visualizer and some interpretation of the above plot.


    API Reference
    -------------

    .. automodule:: yellowbrick.regressor.mymodule
        :members: MyVisualizer
        :undoc-members:
        :show-inheritance:

This is a pretty good structure for a documentation page; a brief introduction followed by a code example with a visualization included using `the plot directive <https://matplotlib.org/devel/plot_directive.html>`_. This will render the ``MyVisualizer`` image in the document along with links for the complete source code, the png, and the pdf versions of the image. It will also have the "alt-text" (for screen-readers) and will not display the source because of the ``:include-source:`` option. If ``:include-source:`` is omitted, the source will be included by default.

The primary section is wrapped up with a discussion about how to interpret the visualizer and use it in practice. Finally the ``API Reference`` section will use ``automodule`` to include the documentation from your docstring.

At this point there are several places where you can list your visualizer, but to ensure it is included in the documentation it *must be listed in the TOC of the local index*. Find the ``index.rst`` file in your subdirectory and add your rst file (without the ``.rst`` extension) to the ``..toctree::`` directive. This will ensure the documentation is included when it is built.

Building the Docs
~~~~~~~~~~~~~~~~~

Speaking of, you can build your documentation by changing into the ``docs`` directory and running ``make html``, the documentation will be built and rendered in the ``_build/html`` directory. You can view it by opening ``_build/html/index.html`` then navigating to your documentation in the browser.

There are several other places that you can list your visualizer including:

 - ``docs/index.rst`` for a high level overview of our visualizers
 - ``DESCRIPTION.rst`` for inclusion on PyPI
 - ``README.md`` for inclusion on GitHub

Please ask for the maintainer's advice about how to include your visualizer in these pages.


Generating the Gallery
~~~~~~~~~~~~~~~~~~~~~~

In v1.0, we have adopted Matplotlib's `plot directive <https://matplotlib.org/devel/plot_directive.html>`_ which means that the majority of the images generated for the documentation are generated automatically. One exception is the gallery; the images for the gallery must still be generated manually.

If you have contributed a new visualizer as described in the above section, please also add it to the gallery, both to docs/gallery.py and to docs/gallery.rst. (Make sure you have already installed Yellowbrick in editable mode, from the top level directory: pip install -e .)

If you want to regenerate a single image (e.g. the elbow curve plot), you can do so as follows: ::

    $ python docs/gallery.py elbow

If you want to regenerate them all (note: this takes a long time!) ::

    $ python docs/gallery.py all
