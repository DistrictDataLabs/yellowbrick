# Contributing to Yellowbrick

**NOTE: This document is a "getting started" summary for contributing to the Yellowbrick project.** To read the full contributor's guide, please visit the [contributing page](http://www.scikit-yb.org/en/latest/about.html#contributing) in the documentation. Please make sure to read this page carefully to ensure the review process is as smooth as possible and to ensure the greatest likelihood of having your contribution be merged.

## How to Contribute

If you would like to contribute to Yellowbrick, you can do so in the following ways:

- Add issues or bugs to the bug tracker: [https://github.com/DistrictDataLabs/yellowbrick/issues](https://github.com/DistrictDataLabs/yellowbrick/issues)
- Work on a card on the dev board: [https://waffle.io/DistrictDataLabs/yellowbrick](https://waffle.io/DistrictDataLabs/yellowbrick)
- Create a pull request in Github: [https://github.com/DistrictDataLabs/yellowbrick/pulls](https://github.com/DistrictDataLabs/yellowbrick/pulls)

Here's how to get set up with Yellowbrick in development mode (since the project is still under development).

1. Fork and clone the repository. After clicking fork in the upper right corner for your own copy of Yellowbrick to your github account. Clone it in a directory of your choice.

        $ git clone https://github.com/[YOURUSERNAME]/yellowbrick
        $ cd yellowbrick

2. Create virtualenv and create the dependencies.

        $ virtualenv venv
        $ source venv/bin/activate
        $ pip install -r requirements.txt

3. Fetch and switch to development.

        $ git fetch
        $ git checkout develop

## Git Conventions

The Yellowbrick repository is set up in a typical production/release/development cycle as described in "[A Successful Git Branching Model](http://nvie.com/posts/a-successful-git-branching-model/)." A typical workflow is as follows:

1. Select a card from the [dev board](https://waffle.io/DistrictDataLabs/yellowbrick), preferably one that is "ready" then move it to "in-progress".

2. Create a branch off of develop called "feature-[feature name]", work and commit into that branch.

        $ git checkout -b feature-myfeature develop

3. Create a Pull Request from your feature branch to the Yellowbrick _develop branch_ in order to  discuss the issue with the core developers. Every commit to your feature branch will automatically be added to the PR.

4. You have two choices to commit your feature back to us.

    a. If you are a core contributor, once you are done working (and everything is tested) merge your feature into develop.

        $ git checkout develop
        $ git merge --no-ff feature-myfeature
        $ git branch -d feature-myfeature
        $ git push origin develop

    b. Otherwise submit a pull request _to the develop branch_, we will review it and merge it if it passes all the requirements.

5. Repeat. Releases will be routinely pushed into master via release branches, then deployed to the server.

## Throughput

[![Throughput Graph](https://graphs.waffle.io/DistrictDataLabs/yellowbrick/throughput.svg)](https://waffle.io/DistrictDataLabs/yellowbrick/metrics/throughput)
