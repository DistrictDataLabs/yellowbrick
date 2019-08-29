.. -*- mode: rst -*-

Getting Started on GitHub
=========================

Yellowbrick is hosted on GitHub at https://github.com/DistrictDataLabs/yellowbrick.

The typical workflow for a contributor to the codebase is as follows:

1. **Discover** a bug or a feature by using Yellowbrick.
2. **Discuss** with the core contributors by `adding an issue <https://github.com/DistrictDataLabs/yellowbrick/issues>`_.
3. **Fork** the repository into your own GitHub account.
4. Create a **Pull Request** first thing to `connect with us <https://github.com/DistrictDataLabs/yellowbrick/pulls>`_ about your task.
5. **Code** the feature, write the tests and documentation, add your contribution.
6. **Review** the code with core contributors who will guide you to a high quality submission.
7. **Merge** your contribution into the Yellowbrick codebase.

We believe that *contribution is collaboration* and therefore emphasize *communication* throughout the open source process. We rely heavily on GitHub's social coding tools to allow us to do this. For instance, we use GitHub's `milestone <https://help.github.com/en/articles/about-milestones>`_ feature to focus our development efforts for each Yellowbrick semester, so be sure to check out the issues associated with our `current milestone <https://github.com/districtdatalabs/yellowbrick/milestones>`_!

Once you have a good sense of how you are going to implement the new feature (or fix the bug!), you can reach out for feedback from the maintainers by creating a `pull request <https://github.com/DistrictDataLabs/yellowbrick/pulls>`_. Ideally, any pull request should be capable of resolution within 6 weeks of being opened. This timeline helps to keep our pull request queue small and allows Yellowbrick to maintain a robust release schedule to give our users the best experience possible. However, the most important thing is to keep the dialogue going! And if you're unsure whether you can complete your idea within 6 weeks, you should still go ahead and open a PR and we will be happy to help you scope it down as needed.

If we have comments or questions when we evaluate your pull request and receive no response, we will also close the PR after this period of time. Please know that this does not mean we don't value your contribution, just that things go stale. If in the future you want to pick it back up, feel free to address our original feedback and to reference the original PR in a new pull request. 

.. note:: Please note that if we feel your solution has not been thought out in earnest, or if the PR is not aligned with our `current milestone <https://github.com/districtdatalabs/yellowbrick/milestones>`_ goals, we may reach out to ask that you close the PR so that we can prioritize reviewing the most critical feature requests and bug fixes.

Forking the Repository
----------------------

The first step is to fork the repository into your own account. This will create a copy of the codebase that you can edit and write to. Do so by clicking the **"fork"** button in the upper right corner of the Yellowbrick GitHub page.

Once forked, use the following steps to get your development environment set up on your computer:

1. Clone the repository.

    After clicking the fork button, you should be redirected to the GitHub page of the repository in your user account. You can then clone a copy of the code to your local machine.::

        $ git clone https://github.com/[YOURUSERNAME]/yellowbrick
        $ cd yellowbrick

2. Create a virtual environment.

    Yellowbrick developers typically use `virtualenv <https://virtualenv.pypa.io/en/stable/>`_ (and `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_), `pyenv <https://github.com/pyenv/pyenv-virtualenv>`_ or `conda envs <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ in order to manage their Python version and dependencies. Using the virtual environment tool of your choice, create one for Yellowbrick. Here's how with virtualenv::

        $ virtualenv venv

    To develop with a conda environment, the conda-forge channel is needed to install some testing dependencies. The following command adds the channel with the highest priority::

        $ conda config --add channels conda-forge

3. Install dependencies.

    Yellowbrick's dependencies are in the ``requirements.txt`` document at the root of the repository. Open this file and uncomment any dependencies marked as for development only. Then install the package in editable mode::

        $ pip install -e .

    This will add Yellowbrick to your PYTHONPATH so that you don't need to reinstall it each time you make a change during development.
    
    Note that there may be other dependencies required for development and testing; you can simply install them with ``pip``. For example to install
    the additional dependencies for building the documentation or to run the
    test suite, use the ``requirements.txt`` files in those directories::

        $ pip install -r tests/requirements.txt
        $ pip install -r docs/requirements.txt

4. Switch to the develop branch.

    The Yellowbrick repository has a ``develop`` branch that is the primary working branch for contributions. It is probably already the branch you're on, but you can make sure and switch to it as follows::

        $ git fetch
        $ git checkout develop

At this point you're ready to get started writing code. 

Branching Convention
--------------------

The Yellowbrick repository is set up in a typical production/release/development cycle as described in "`A Successful Git Branching Model <http://nvie.com/posts/a-successful-git-branching-model/>`_." The primary working branch is the ``develop`` branch. This should be the branch that you are working on and from, since this has all the latest code. The ``master`` branch contains the latest stable version and release_, which is pushed to PyPI_. No one but core contributors will generally push to master.

You should work directly in your fork. In order to reduce the number of merges (and merge conflicts) we kindly request that you utilize a feature branch off of ``develop`` to work in::

    $ git checkout -b feature-myfeature develop

We also recommend setting up an ``upstream`` remote so that you can easily pull the latest development changes from the main Yellowbrick repository (see `configuring a remote for a fork <https://help.github.com/articles/configuring-a-remote-for-a-fork/>`_). You can do that as follows::

    $ git remote add upstream https://github.com/DistrictDataLabs/yellowbrick.git
    $ git remote -v
    origin    https://github.com/YOUR_USERNAME/YOUR_FORK.git (fetch)
    origin    https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)
    upstream  https://github.com/DistrictDataLabs/yellowbrick.git (fetch)
    upstream  https://github.com/DistrictDataLabs/yellowbrick.git (push)

When you're ready, request a code review for your pull request. 

Pull Requests
-------------

A `pull request (PR) <https://help.github.com/articles/about-pull-requests/>`_ is a GitHub tool for initiating an exchange of code and creating a communication channel for Yellowbrick maintainers to discuss your contribution. In essenence, you are requesting that the maintainers merge code from your forked repository into the develop branch of the primary Yellowbrick repository. Once completed, your code will be part of Yellowbrick!

When starting a Yellowbrick contribution, *open the pull request as soon as possible*. We use your PR issue page to discuss your intentions and to give guidance and direction. Every time you push a commit into your forked repository, the commit is automatically included with your pull request, therefore we can review as you code. The earlier you open a PR, the more easily we can incorporate your updates, we'd hate for you to do a ton of work only to discover someone else already did it or that you went in the wrong direction and need to refactor.

.. note:: For a great example of a pull request for a new feature visualizer, check out `this one <https://github.com/DistrictDataLabs/yellowbrick/pull/232>`_ by `Carlo Morales <https://github.com/cjmorale>`_.

Opening a Pull Request
~~~~~~~~~~~~~~~~~~~~~~

When you open a pull request, ensure it is from your forked repository to the develop branch of `github.com/districtdatalabs/yellowbrick <https://github.com/districtdatalabs/yellowbrick>`_; we will not merge a PR into the master branch. Title your Pull Request so that it is easy to understand what you're working on at a glance. Also be sure to include a reference to the issue that you're working on so that correct references are set up.

.. note:: All pull requests should be into the ``yellowbrick/develop`` branch from your forked repository.

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

Then we will "Squash and Merge" your contribution, combining all of your commits into a single commit and merging it into the ``develop`` branch of Yellowbrick. Congratulations! Once your contribution has been merged into master, you will be officially listed as a contributor.

After Your Pull Request is Merged
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After your pull request is merged, you should update your local fork, either by pulling from ``upstream`` ``develop``::

    $ git checkout develop
    $ git pull upstream develop
    $ git push origin develop

or by manually merging your feature into your fork's ``develop`` branch.::

    $ git checkout develop
    $ git merge --no-ff feature-myfeature
    $ git push origin develop

Then you can safely delete the old feature branch, both locally and on GitHub. Now head back to `the backlog <https://github.com/districtdatalabs/yellowbrick/issues>`_ and checkout another issue!

.. _release: https://github.com/DistrictDataLabs/yellowbrick/releases
.. _PyPI: https://pypi.python.org/pypi/yellowbrick