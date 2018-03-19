.. -*- mode: rst -*-

Contributing
============

Yellowbrick is an open source project that is supported by a community who will gratefully and humbly accept any contributions you might make to the project. Large or small, any contribution makes a big difference; and if you've never contributed to an open source project before, we hope you will start with Yellowbrick!

Principally, Yellowbrick development is about the addition and creation of *visualizers* --- objects that learn from data and create a visual representation of the data or model. Visualizers integrate with scikit-learn estimators, transformers, and pipelines for specific purposes and as a result, can be simple to build and deploy. The most common contribution is a new visualizer for a specific model or model family. We'll discuss in detail how to build visualizers later.

Beyond creating visualizers, there are many ways to contribute:

- Submit a bug report or feature request on `GitHub Issues <https://github.com/DistrictDataLabs/yellowbrick/issues>`_.
- Contribute an Jupyter notebook to our `examples gallery <https://github.com/DistrictDataLabs/yellowbrick/tree/develop/examples>`_.
- Assist us with `user testing <http://www.scikit-yb.org/en/latest/evaluation.html>`_.
- Add to the documentation or help with our website, `scikit-yb.org <http://www.scikit-yb.org>`_
- Write unit or integration tests for our project.
- Answer questions on our issues, mailing list, Stack Overflow, and elsewhere.
- Translate our documentation into another language.
- Write a blog post, tweet, or share our project with others.
- Teach someone how to use Yellowbrick.

As you can see, there are lots of ways to get involved and we would be very happy for you to join us! The only thing we ask is that you abide by the principles of openness, respect, and consideration of others as described in the `Python Software Foundation Code of Conduct <https://www.python.org/psf/codeofconduct/>`_.

Getting Started on GitHub
-------------------------

Yellowbrick is hosted on GitHub at https://github.com/DistrictDataLabs/yellowbrick.

The typical workflow for a contributor to the codebase is as follows:

1. **Discover** a bug or a feature by using Yellowbrick.
2. **Discuss** with the core contributors by `adding an issue <https://github.com/DistrictDataLabs/yellowbrick/issues>`_.
3. **Assign** yourself the task by pulling a card from our `Waffle Kanban <https://waffle.io/DistrictDataLabs/yellowbrick>`_
4. **Fork** the repository into your own GitHub account.
5. Create a **Pull Request** first thing to `connect with us <https://github.com/DistrictDataLabs/yellowbrick/pulls>`_ about your task.
6. **Code** the feature, write the tests and documentation, add your contribution.
7. **Review** the code with core contributors who will guide you to a high quality submission.
8. **Merge** your contribution into the Yellowbrick codebase.

.. note:: Please create a pull request as soon as possible, even before you've started coding. This will allow the core contributors to give you advice about where to add your code or utilities and discuss other style choices and implementation details as you go. Don't wait!

We believe that *contribution is collaboration* and therefore emphasize *communication* throughout the open source process. We rely heavily on GitHub's social coding tools to allow us to do this.

Forking the Repository
~~~~~~~~~~~~~~~~~~~~~~

The first step is to fork the repository into your own account. This will create a copy of the codebase that you can edit and write to. Do so by clicking the **"fork"** button in the upper right corner of the Yellowbrick GitHub page.

Once forked, use the following steps to get your development environment set up on your computer:

1. Clone the repository.

    After clicking the fork button, you should be redirected to the GitHub page of the repository in your user account. You can then clone a copy of the code to your local machine.::

        $ git clone https://github.com/[YOURUSERNAME]/yellowbrick
        $ cd yellowbrick

2. Create a virtual environment.

    Yellowbrick developers typically use `virtualenv <https://virtualenv.pypa.io/en/stable/>`_ (and `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_), `pyenv <https://github.com/pyenv/pyenv-virtualenv>`_ or `conda envs <https://conda.io/docs/using/envs.html>`_ in order to manage their Python version and dependencies. Using the virtual environment tool of your choice, create one for Yellowbrick. Here's how with virtualenv::

        $ virtualenv venv

3. Install dependencies.

    Yellowbrick's dependencies are in the ``requirements.txt`` document at the root of the repository. Open this file and uncomment the dependencies that are for development only. Then install the dependencies with ``pip``::

        $ pip install -r requirements.txt

    Note that there may be other dependencies required for development and testing; you can simply install them with ``pip``. For example to install
    the additional dependencies for building the documentation or to run the
    test suite, use the ``requirements.txt`` files in those directories::

        $ pip install -r tests/requirements.txt
        $ pip install -r docs/requirements.txt

4. Switch to the develop branch.

    The Yellowbrick repository has a ``develop`` branch that is the primary working branch for contributions. It is probably already the branch you're on, but you can make sure and switch to it as follows::

        $ git fetch
        $ git checkout develop

At this point you're ready to get started writing code. If you're going to take on a specific task, we'd strongly encourage you to check out the issue on `Waffle <https://waffle.io/DistrictDataLabs/yellowbrick>`_ and create a `pull request <https://github.com/DistrictDataLabs/yellowbrick/pulls>`_ *before you start coding* to better foster communication with other contributors. More on this in the next section.

Pull Requests
~~~~~~~~~~~~~

A `pull request (PR) <https://help.github.com/articles/about-pull-requests/>`_ is a GitHub tool for initiating an exchange of code and creating a communication channel for Yellowbrick maintainers to discuss your contribution. In essenence, you are requesting that the maintainers merge code from your forked repository into the develop branch of the primary Yellowbrick repository. Once completed, your code will be part of Yellowbrick!

When starting a Yellowbrick contribution, *open the pull request as soon as possible*. We use your PR issue page to discuss your intentions and to give guidance and direction. Every time you push a commit into your forked repository, the commit is automatically included with your pull request, therefore we can review as you code. The earlier you open a PR, the more easily we can incorporate your updates, we'd hate for you to do a ton of work only to discover someone else already did it or that you went in the wrong direction and need to refactor.

.. note:: For a great example of a pull request for a new feature visualizer, check out `this one <https://github.com/DistrictDataLabs/yellowbrick/pull/232>`_ by `Carlo Morales <https://github.com/cjmorale>`_.

When you open a pull request, ensure it is from your forked repository to the develop branch of `github.com/districtdatalabs/yellowbrick <https://github.com/districtdatalabs/yellowbrick>`_; we will not merge a PR into the master branch. Title your Pull Request so that it is easy to understand what you're working on at a glance. Also be sure to include a reference to the issue that you're working on so that correct references are set up.

After you open a PR, you should get a message from one of the maintainers. Use that time to discuss your idea and where best to implement your work. Feel free to go back and forth as you are developing with questions in the comment thread of the PR. Once you are ready, please ensure that you explicitly ping the maintainer to do a code review. Before code review, your PR should contain the following:

1. Your code contribution
2. Tests for your contribution
3. Documentation for your contribution
4. A PR comment describing the changes you made and how to use them
5. A PR comment that includes an image/example of your visualizer

At this point your code will be formally reviewed by one of the contributors. We use GitHub's code review tool, starting a new code review and adding comments to specific lines of code as well as general global comments. Please respond to the comments promptly, and don't be afraid to ask for help implementing any requested changes! You may have to go back and forth a couple of times to complete the code review.

When the following is true:

1. Code is reviewed by at least one maintainer
2. Continuous Integration tests have passed
3. Code coverage and quality have not decreased
4. Code is up to date with the yellowbrick develop branch

Then we will "Squash and Merge" your contribution, combining all of your commits into a single commit and merging it into the develop branch of Yellowbrick. Congratulations! Once your contribution has been merged into master, you will be officially listed as a contributor.

Developing Visualizers
----------------------

In this section, we'll discuss the basics of developing visualizers. This of course is a big topic, but hopefully these simple tips and tricks will help make sense. First thing though, check out this presentation that we put together on yellowbrick development, it discusses the expected user workflow, our integration with scikit-learn, our plans and roadmap, etc:

.. raw:: html

    <iframe src="https://www.slideshare.net/BenjaminBengfort/slideshelf" width="615px" height="470px" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:none;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe>

One thing that is necessary is a good understanding of scikit-learn and Matplotlib. Because our API is intended to integrate with scikit-learn, a good start is to review `"APIs of scikit-learn objects" <http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects>`_ and `"rolling your own estimator" <http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator>`_. In terms of matplotlib, use Yellowbrick's guide :doc:`matplotlib`. Additional resources include `Nicolas P. Rougier's Matplotlib tutorial <https://www.labri.fr/perso/nrougier/teaching/matplotlib/>`_ and `Chris Moffitt's Effectively Using Matplotlib <http://pbpython.com/effective-matplotlib.html>`_.

Visualizer API
~~~~~~~~~~~~~~

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

    import matplotlib.pyplot as plot

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

            self.ax.plot(X)

        def finalize(self):
            self.set_title("My Visualizer")

This simple visualizer simply draws a line graph for some input dataset X, intersecting with the scikit-learn API at the ``fit()`` method. A user would use this visualizer in the typical style::

    visualizer = MyVisualizer()
    visualizer.fit(X)
    visualizer.poof()

Score visualizers work on the same principle but accept an additional required ``model`` argument. Score visualizers wrap the model (which can be either instantiated or uninstantiated) and then pass through all attributes and methods through to the underlying model, drawing where necessary.

Testing
~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~

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

Documentation
~~~~~~~~~~~~~

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

Advanced Development
--------------------

In this section we discuss more advanced contributing guidelines including setting up branches for development as well as the release cycle. This section is intended for maintainers and core contributors of the Yellowbrick project. If you would like to be a maintainer please contact one of the current maintainers of the project.

Branching Convention
~~~~~~~~~~~~~~~~~~~~

The Yellowbrick repository is set up in a typical production/release/development cycle as described in "`A Successful Git Branching Model <http://nvie.com/posts/a-successful-git-branching-model/>`_." The primary working branch is the ``develop`` branch. This should be the branch that you are working on and from, since this has all the latest code. The ``master`` branch contains the latest stable version and release_, which is pushed to PyPI_. No one but core contributors will generally push to master.

.. note:: All pull requests should be into the ``yellowbrick/develop`` branch from your forked repository.

You can work directly in your fork and create a pull request from your fork's develop branch into ours. We also recommend setting up an ``upstream`` remote so that you can easily pull the latest development changes from the main Yellowbrick repository (see `configuring a remote for a fork <https://help.github.com/articles/configuring-a-remote-for-a-fork/>`_). You can do that as follows::

    $ git remote add upstream https://github.com/DistrictDataLabs/yellowbrick.git
    $ git remote -v
    origin    https://github.com/YOUR_USERNAME/YOUR_FORK.git (fetch)
    origin    https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)
    upstream  https://github.com/DistrictDataLabs/yellowbrick.git (fetch)
    upstream  https://github.com/DistrictDataLabs/yellowbrick.git (push)

When you're ready, request a code review for your pull request. Then, when reviewed and approved, you can merge your fork into our main branch. Make sure to use the "Squash and Merge" option in order to create a Git history that is understandable.

.. note:: When merging a pull request, use the "squash and merge" option.

Core contributors have write access to the repository. In order to reduce the number of merges (and merge conflicts) we recommend that you utilize a feature branch off of develop to do intermediate work in::

    $ git checkout -b feature-myfeature develop

Once you are done working (and everything is tested) merge your feature into develop.::

    $ git checkout develop
    $ git merge --no-ff feature-myfeature
    $ git branch -d feature-myfeature
    $ git push origin develop

Head back to Waffle and checkout another issue!

Releases
~~~~~~~~

When ready to create a new release we branch off of develop as follows::

    $ git checkout -b release-x.x

This creates a release branch for version x.x. At this point do the version bump by modifying ``version.py`` and the test version in ``tests/__init__.py``. Make sure all tests pass for the release and that the documentation is up to date. There may be style changes or deployment options that have to be done at this phase in the release branch. At this phase you'll also modify the ``changelog`` with the features and changes in the release.

Once the release is ready for prime-time, merge into master::

    $ git checkout master
    $ git merge --no-ff --no-edit release-x.x

Tag the release in GitHub::

    $ git tag -a vx.x
    $ git push origin vx.x

You'll have to go to the release_ page to edit the release with similar information as added to the changelog. Once done, push the release to PyPI::

    $ make build
    $ make deploy

Check that the PyPI page is updated with the correct version and that ``pip install -U yellowbrick`` updates the version and works correctly. Also check the documentation on PyHosted, ReadTheDocs, and on our website to make sure that it was correctly updated. Finally merge the release into develop and clean up::

    $ git checkout develop
    $ git merge --no-ff --no-edit release-x.x
    $ git branch -d release-x.x

Hotfixes and minor releases also follow a similar pattern; the goal is to effectively get new code to users as soon as possible!

.. _release: https://github.com/DistrictDataLabs/yellowbrick/releases
.. _PyPI: https://pypi.python.org/pypi/yellowbrick
