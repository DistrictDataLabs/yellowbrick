# Yellowbrick Examples

[![Visualizers](../docs/images/readme/banner.png)](../docs/images/readme/banner.png)

Welcome to the yellowbrick examples directory! This directory contains a gallery of visualizers and their application to classification, regression, clustering, and other machine learning techniques with scikit-learn. Examples have been submitted both by the Yellowbrick team and also users like you! The result is a rich gallery of tools and techniques to equip your machine learning with visual diagnostics and visualizer workflows!

## Getting Started

The notebook to explore first is the `examples.ipynb` Jupyter notebook. This notebook contains the executable examples from the tutorial in the documentation. You can run the notebook as follows:

```
$ jupyter notebook examples.ipynb
```

If you don't have jupyter installed, or other dependencies, you may have to `pip install` them.

## Organization

The examples directory contains many notebooks, folders and files. At the top level you will see the following:

- examples.ipynb: a notebook with executable versions of the tutorial visualizers
- palettes.ipynb: a visualization of the Yellowbrick palettes
- regression.ipynb: a notebook exploring the regression model visualizers.

In addition to these files and directory, you will see many other directories, whose names are the GitHub usernames of their contributors. You can explore these user submitted examples or submit your own!

### Contributing

To contribute an example notebook of your own, perform the following steps:

1. Fork the repository into your own account
2. Checkout the develop branch (see [contributing to Yellowbrick](http://www.scikit-yb.org/en/latest/about.html#contributing) for more.
3. Create a directory in the repo, `examples/username` where username is your GitHub username.
4. Create a notebook in that directory with your example. See [user testing](http://www.scikit-yb.org/en/latest/evaluation.html) for more.
5. Commit your changes back to your fork.
6. Submit a pull-request from your develop branch to the Yellowbrick develop branch.
7. Complete the code review steps with a Yellowbrick team member.

That's it -- thank you for contributing your example!

A couple of notes. First, please make sure that the Jupyter notebook you submit is "run" -- that is it has the output saved to the notebook and is viewable on GitHub (empty notebooks don't serve well as a gallery). Second, please do not commit datasets, but instead provide instructions for downloading the dataset. You can create a downloader utility similar to ours.

One great tip, is to create your PR right after you fork the repo; that way we can work with you on the changes you're making and communicate about how to have a very successful contribution!
