.. -*- mode: rst -*-

Advanced Development Topics
===========================

In this section we discuss more advanced contributing guidelines such as code conventions,the release life cycle or branch management. This section is intended for maintainers and core contributors of the Yellowbrick project. If you would like to be a maintainer please contact one of the current maintainers of the project.

Visualizer Review Checklist
---------------------------

As the visualizer API has matured over time, we've realized that there are a number of routine items that must be in place to consider a visualizer truly complete and ready for prime time. This list is also extremely helpful for reviewing code submissions to ensure that visualizers are consistently implemented, tested, and documented. Though we do not expect these items to be checked off on every PR, the below list includes some guidance about what to look for when reviewing or writing a new Visualizer.

.. note:: The ``contrib`` module is a great place for work-in-progress Visualizers!

Code Conventions
~~~~~~~~~~~~~~~~

- Ensure the visualizer API is met.

    The basic principle of the visualizer API is that scikit-learn methods such as ``fit()``, ``transform()``, ``score()``, etc. perform interactions with scikit-learn or other computations and call the ``draw()`` method. Calls to matplotlib should happen only in ``draw()`` or ``finalize()``.

- Create a quick method for the visualizer.

    In addition to creating the visualizer class, ensure there is an associated quick method that returns the visualizer and creates the visualization in one line of code!

- Subclass the correct visualizer.

    Ensure that the visualizer is correctly subclassed in the class hierarchy. If you're not sure what to subclass, please ping a maintainer, they'd be glad to help!

- Ensure numpy array comparisons are not ambiguous.

    Often there is code such as ``if y:`` where ``y`` is an array. However this is ambiguous when used with numpy arrays and other data containers. Change this code to ``y is not None`` or ``len(y) > 0`` or use ``np.all`` or ``np.any`` to test if the contents of the array are truthy/falsy.

- Add ``random_state`` argument to visualizer.

    If the visualizer uses/wraps a utility that also has ``random_state``, then the visualizer itself needs to also have this argument which defaults to ``None`` and is passed to all internal stochastic behaviors. This ensures that image comparison testing will work and that users can get repeated behavior from visualizers.

- Use ``np.unique`` instead of `set`.

    If you need the unique values from a list or array, we prefer to use numpy methods wherever possible. We performed some limited benchmarking and believe that ``np.unique`` is a bit faster and more efficient.

- Use sklearn underscore suffix for learned parameters.

    Any parameters that are learned during ``fit()`` should only be added to the visualizer when ``fit()`` is called (this is also how we determine if a visualizer is fitted or not) and should be identified with an underscore suffix. For example, in classification visualizers, the classes can be either passed in by the user or determined when they are passed in via fit, therefore it should be ``self.classes_``. This is also true for other learned parameters, e.g. ``self.score_``, even though this is not created during ``fit()``.

- Correctly set the title in finalize.

    Use the ``self.set_title()`` method to set a default title; this allows the user to specify a custom title in the initialization arguments.

Testing Conventions
~~~~~~~~~~~~~~~~~~~

- Ensure there is an image comparison test.

    Ensure there is at least one image comparison test per visualizer. This is the primary regression testing of Yellowbrick and these tests catch a lot when changes occur in our dependencies or environment.

- Use pytest assertions rather than unittest.

    We prefer ``assert 2+2 == 4`` rather than ``self.assertEquals(2+2, 4)``. Though there is a lot of legacy unittest assertions, we've moved to pytest and one day believe we will have removed the unittest dependency.

- Use test fixtures and sklearn dataset generators.

    Data is the key to testing with Yellowbrick; often the test package will have fixtures in ``conftest.py`` that can be directly used (e.g. binary vs. multiclass in the ``test_classifier`` package). If one isn't available feel free to use randomly generated datasets from the ``sklearn.datasets`` module e.g. ``make_classification``, ``make_regression``, or ``make_blobs``. For integration testing, please feel free to use one of the Yellowbrick datasets.

- Fix all ``random_state`` arguments.

    Be on the lookout for any method (particularly sklearn methods) that have a ``random_state`` argument and be sure to fix them so that tests always pass!

- Test a variety of inputs.

    Machine learning can be done on a variety of inputs for ``X`` and ``y``, ensure there is a test with numpy arrays, pandas DataFrame and Series objects, and with Python lists.

- Test that ``fit()`` returns self.

    When doing end-to-end testing, we like to ``assert oz.fit() is oz`` to ensure the API is maintained.

- Test that ``score()`` between zero and one.

    With visualizers that have a ``score()`` method, we like to ``assert 0.0 <= oz.score() >=1.0`` to ensure the API is maintained.

Documentation Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~

- Visualizer DocString is correct.

    The visualizer docstring should be present under the class and contain a narrative about the visualizer and its arguments with the numpydoc style.

- API Documentation.

    All visualizers should have their own API page under ``docs/api/[yb-module]``. This documentation should include an ``automodule`` statement. Generally speaking there is also an image generation script of the same name in this folder so that the documentation images can be generated on demand.

- Listing the visualizer.

    The visualizer should be listed in a number of places including: ``docs/api/[yb-module]/index.rst``, ``docs/api/index.rst``, ``docs/index.rst``, ``README.md``, and ``DESCRIPTION.rst``.

- Include a gallery image.

    Please also add the visualizer image to the gallery!

- Update added to the changelog.

    To reduce the time it takes to put together the changelog, we'd like to update it when we add new features and visualizers rather than right before the release.

Branching Convention
--------------------

The Yellowbrick repository is set up in a typical production/release/development cycle as described in "`A Successful Git Branching Model <http://nvie.com/posts/a-successful-git-branching-model/>`_." The primary working branch is the ``develop`` branch. This should be the branch that you are working on and from, since this has all the latest code. The ``master`` branch contains the latest stable version and release_, which is pushed to PyPI_. No one but core contributors will generally push to master.

.. note:: All pull requests should be into the ``yellowbrick/develop`` branch from your forked repository.

You can work directly in your fork and create a pull request from your fork's ``develop`` branch into ours. In order to reduce the number of merges (and merge conflicts) we recommend that you utilize a feature branch off of ``develop`` to work in::

    $ git checkout -b feature-myfeature develop

We also recommend setting up an ``upstream`` remote so that you can easily pull the latest development changes from the main Yellowbrick repository (see `configuring a remote for a fork <https://help.github.com/articles/configuring-a-remote-for-a-fork/>`_). You can do that as follows::

    $ git remote add upstream https://github.com/DistrictDataLabs/yellowbrick.git
    $ git remote -v
    origin    https://github.com/YOUR_USERNAME/YOUR_FORK.git (fetch)
    origin    https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)
    upstream  https://github.com/DistrictDataLabs/yellowbrick.git (fetch)
    upstream  https://github.com/DistrictDataLabs/yellowbrick.git (push)

When you're ready, request a code review for your pull request. Then, when reviewed and approved, we will merge the branch of your fork into our ``develop`` branch. 


Merging Pull Requests
~~~~~~~~~~~~~~~~~~~~~

Our convention is that the person who performs the code review should merge the pull request (since reviewing is hard work and deserves due credit!). Only core contributors have write access to the repository and can merge pull requests. Some preferences for commit messages when merging in pull requests:

- Make sure to use the "Squash and Merge" option in order to create a Git history that is understandable. 
- Keep the title of the commit short and descriptive; be sure it includes the PR #.
- Craft a commit message body that is 1-3 sentences, depending on the complexity of the commit; it should explicitly reference any issues being closed or opened using `GitHub's commit message keywords <https://help.github.com/articles/closing-issues-using-keywords/>`_.

.. note:: When merging a pull request, use the "squash and merge" option.


Releases
--------

When ready to create a new release we branch off of develop as follows::

    $ git checkout -b release-x.x

This creates a release branch for version x.x. At this point do the version bump by modifying ``version.py`` and the test version in ``tests/__init__.py``. Make sure all tests pass for the release and that the documentation is up to date. Note, to build the docs see the :ref:`documentation notes <documentation>`. There may be style changes or deployment options that have to be done at this phase in the release branch. At this phase you'll also modify the ``changelog`` with the features and changes in the release that have not already been marked.

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