.. -*- mode: rst -*-

Advanced Development Topics
===========================

In this section we discuss more advanced contributing guidelines such as code conventions,the release life cycle or branch management. This section is intended for maintainers and core contributors of the Yellowbrick project. If you would like to be a maintainer please contact one of the current maintainers of the project.

Reviewing Pull Requests
-----------------------

We use several strategies when reviewing pull requests from contributors to Yellowbrick. If the pull request affects only a single file or a small portion of the code base, it is sometimes sufficient to review the code using `GitHub's lightweight code review feature <https://github.com/features/code-review/>`_. However, if the changes impact a number of files or modify the documentation, our convention is to add the contributor's fork as a remote, pull, and check out their feature branch locally. From inside your fork of Yellowbrick, this can be done as follows::

    $ git remote add contribsusername https://github.com/contribsusername/yellowbrick.git
    $ git fetch contribsusername
    $ git checkout -b contribsfeaturebranch contribsusername/contribsfeaturebranch

This will allow you to inspect their changes, run the tests, and build the docs locally. If the contributor has elected to allow reviewers to modify their feature branch, you will also be able to push changes directly to their branch::

    $ git add filethatyouchanged.py
    $ git commit -m "Adjusted tolerance levels to appease AppVeyor"
    $ git push contribsusername contribsfeaturebranch

These changes will automatically go into the pull request, which can be useful for making small modifications (e.g. visual test tolerance levels) to get the PR over the finish line.


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

- Use pytest assertions rather than ``unittest.TestCase`` methods.

    We prefer ``assert 2+2 == 4`` rather than ``self.assertEquals(2+2, 4)``. As a result, test classes should not extend ``unittest.Testcase`` but should extend the ``VisualTestCase`` in the tests package. Note that if you're writing tests that do not generate matplotlib figures you can simply extend ``object``.

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

Merging Pull Requests
---------------------

Our convention is that the person who performs the code review should merge the pull request (since reviewing is hard work and deserves due credit!). Only core contributors have write access to the repository and can merge pull requests. Some preferences for commit messages when merging in pull requests:

- Make sure to use the "Squash and Merge" option in order to create a Git history that is understandable.
- Keep the title of the commit short and descriptive; be sure it includes the PR #.
- Craft a commit message body that is 1-3 sentences, depending on the complexity of the commit; it should explicitly reference any issues being closed or opened using `GitHub's commit message keywords <https://help.github.com/articles/closing-issues-using-keywords/>`_.

.. note:: When merging a pull request, use the "squash and merge" option.


Releases
--------

To ensure we get new code to our users as soon and as bug free as possible we periodically create major, minor, and hotfix version releases that are merged from the ``develop`` branch into ``master`` and pushed to PyPI and Anaconda Cloud. Our release cycle ensures that stable code can be found in the master branch and pip installed and that we can test our development code thoroughly before a release.

.. note:: The following steps must be taken by a maintainer with access to the primary (upstream) Yellowbrick repository. Any reference to ``origin`` refers to github.com/DistrictDataLabs/yellowbrick.

The first step is to create a release branch from develop - this allows us to do "release-work" (e.g. a version bump, changelog stuff, etc.) in a branch that is neither ``develop`` nor ``master`` and to test the release before deployment::

    $ git checkout develop
    $ git pull origin develop
    $ git checkout -b release-x.x

This creates a release branch for version ``x.x`` where ``x`` is a digit. Release versions are described as a number ``x.y.z`` where ``x`` is the major version, ``y`` is the minor version and ``z`` is a patch version. Generally speaking most releases are minor version releases where ``x.y`` becomes ``x.y+1```. Patch versions are infrequent but may also be needed where very little has changed or something quick has to be pushed to fix a critical bug, e.g.g ``x.y`` becomes ``x.y.1``. Major version releases where ``x.y`` become ``x+1.0`` are rare.

At this point do the version bump by modifying ``version.py`` and the test version in ``tests/__init__.py``. Make sure all tests pass for the release and that the documentation is up to date. To build the docs see the :ref:`documentation notes <documentation>`. There may be style changes or deployment options that have to be done at this phase in the release branch. At this phase you'll also modify the ``changelog`` with the features and changes in the release that have not already been marked.

.. note:: Before merging the release to master make sure that the release checklist has been completed!

Once the release is ready for prime-time, merge into master::

    $ git checkout master
    $ git merge --no-ff --no-edit release-x.x
    $ git push origin master

Tag the release in GitHub::

    $ git tag -a vx.x
    $ git push origin vx.x

Now go to the release_ page to convert the tag into a release and add a Markdown version of the changelog notes for those that are accessing the release directly from GitHub.

Deploying to PyPI
~~~~~~~~~~~~~~~~~

Deploying the release to PyPI is fairly straight forward. Ensure that you have valid PyPI login credentials in ``~/.pypirc`` and use the Makefile to deploy as follows::

    $ make build
    $ make deploy

The build process should create ``build`` and ``dist`` directories containing the wheel and source packages as well as a ``.egg-info`` file for deployment. The deploy command registers the version in PyPI and uploads it with Twine.

Deploying to Anaconda Cloud
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These instructions follow the tutorial `"Building conda packages with conda skeleton" <https://conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs-skeleton.html>`_. To deploy release to Anaconda Cloud you first need to have Miniconda or Anaconda installed along with ``conda-build`` and ``anaconda-client`` (which can be installed using ``conda``). Make sure that you run the ``anaconda login`` command using the credentials that allow access to the Yellowbrick channel. If you have an old skeleton directory, make sure to save it with a different name (e.g. yellowbrick.old) before running the skeleton command::

    $ conda skeleton pypi yellowbrick

This should install the latest version of yellowbrick from PyPI - make sure the version matches the expected version of the release! There are some edits that must be made to the ``yellowbrick/meta.yaml`` that is generated as follows::

    about:
        home: http://scikit-yb.org/
        license_file: LICENSE.txt
        doc_url: https://www.scikit-yb.org/en/latest/
        dev_url: https://github.com/DistrictDataLabs/yellowbrick

In addition, you must remove the entire ``test:`` section of the yaml file and add the following to the ``requirements:`` under both ``host:`` and ``run:``. See `example meta.yaml <https://gist.github.com/bbengfort/a77dd0ff610fd10f40926f7426a89486>`_ for a detailed version. Note that the description field in the metadata is pulled from the ``DESCRIPTION.rst`` in the root of the Yellowbrick project. However, Anaconda Cloud requires a Markdown description - the easiest thing to do is to copy it from the existing description.

With the ``meta.yaml`` file setup you can now run the build command for the various Python distributes that Yellowbrick supports::

    $ conda build --python 3.6 yellowbrick
    $ conda build --python 3.7 yellowbrick

After this command completes you should have build files in ``$MINICONDA_HOME/conda-bld/[OS]/yellowbrick-x.x-py3.x_0.tar.bz2``. You can now run conda convert for each of the Python versions using this directory as follows::

    $ conda convert --platform all [path to build] -o $MINICONDA_HOME/conda-bld

At this point you should have builds for all the versions of Python and all platforms Yellowbrick supports. Unfortunately at this point you have to upload them all to Anaconda Cloud::

    $ anaconda upload $MINICONDA_HOME/conda-bld/[OS]/yellowbrick-x.x-py3.x_0.tar.bz2

Once uploaded, the Anaconda Cloud page should reflect the latest version, you may have to edit the description to make sure it's in Markdown format.

Finalizing the Release
~~~~~~~~~~~~~~~~~~~~~~

The last steps in the release process are to check to make sure the release completed successfully. Make sure that the `PyPI page`_ and the `Anaconda Cloud Page`_ are correctly updated to the latest version. Also ensure that ReadTheDocs has correctly built the "latest" documentation on `scikit-yb.org <https://www.scikit-yb.org/en/latest/>`_.

Make sure that you can update the package on your local machine, either in a virtual environment that does not include yellowbrick or in a Python install that is not used for development (e.g. not in the yellowbrick project directory)::

    $ pip install -U yellowbrick
    $ python -c "import yellowbrick; print(yellowbrick.__version__)"

After verifying that the version has been correctly updated you can clean up the project directory::

    $ make clean

After this, it's time to merge the release into develop so that we can get started on the next version! ::

    $ git checkout develop
    $ git merge --no-ff --no-edit release-x.x
    $ git branch -d release-x.x
    $ git push origin develop

Make sure to celebrate the release with the other maintainers and to tweet to everyone to let them know it's time to update Yellowbrick!

.. _release: https://github.com/DistrictDataLabs/yellowbrick/releases
.. _PyPI Page: https://pypi.org/project/yellowbrick/
.. _Anaconda Cloud Page: https://anaconda.org/DistrictDataLabs/yellowbrick
