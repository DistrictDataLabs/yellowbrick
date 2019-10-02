# Yellowbrick Documentation

*Welcome to the Yellowbrick docs!*

If you're looking for information about how to use Yellowbrick, for our contributor's guide, for examples and teaching resources, for answers to frequently asked questions, and more, please visit the latest version of our documentation at [www.scikit-yb.org](https://www.scikit-yb.org/).

## Building the Docs

To build the documents locally, first install the documentation-specific requirements with `pip` using the `requirements.txt` file in the `docs` directory:

```bash
$ pip install -r docs/requirements.txt
```

You will then be able to build the documentation from inside the `docs` directory by running `make html`; the documentation will be built and rendered in the `_build/html` directory. You can view it by opening `_build/html/index.html` then navigating to your documentation in the browser.

## reStructuredText

Yellowbrick uses [Sphinx](http://www.sphinx-doc.org/en/master/index.html) to build our documentation. The advantages of using Sphinx are many; we can more directly link to the documentation and source code of other projects like Matplotlib and scikit-learn using [intersphinx](http://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html). In addition, docstrings used to describe Yellowbrick visualizers can be automatically included when the documentation is built via [autodoc](http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#sphinx.ext.autodoc).

To take advantage of these features, our documentation must be written in reStructuredText (or "rst"). reStructuredText is similar to markdown, but not identical, and does take some getting used to. For instance, styling for things like codeblocks, external hyperlinks, internal cross references, notes, and fixed-width text are all unique in rst.

If you would like to contribute to our documentation and do not have prior experience with rst, we recommend you make use of these resources:

- [A reStructuredText Primer](http://docutils.sourceforge.net/docs/user/rst/quickstart.html)
- [rst notes and cheatsheet](https://cheat.readthedocs.io/en/latest/rst.html)
- [Using the plot directive](https://matplotlib.org/devel/plot_directive.html)

## Adding New Visualizers to the Docs

If you are adding a new visualizer to the docs, there are quite a few examples in the documentation on which you can base your files of similar types.

The primary format for the API section is as follows:

```
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
```

This is a pretty good structure for a documentation page; a brief introduction followed by a code example with a visualization included using [the plot directive](https://matplotlib.org/devel/plot_directive.html). This will render the `MyVisualizer` image in the document along with links for the complete source code, the png, and the pdf versions of the image. It will also have the "alt-text" (for screen-readers) and will not display the source because of the `:include-source:` option. If `:include-source:` is omitted, the source will also be included.

The primary section is wrapped up with a discussion about how to interpret the visualizer and use it in practice. Finally the `API Reference` section will use `automodule` to include the documentation from your docstring.

There are several other places where you can list your visualizer, but to ensure it is included in the documentation it *must be listed in the TOC of the local index*. Find the `index.rst` file in your subdirectory and add your rst file (without the `.rst` extension) to the `..toctree::` directive. This will ensure your documentation is included when it is built.

## Generating the Gallery

In v1.0, we have adopted Matplotlib's [plot directive](https://matplotlib.org/devel/plot_directive.html) which means that the majority of the images generated for the documentation are generated automatically. One exception is the gallery; the images for the gallery must still be generated manually.

If you have contributed a new visualizer as described in the above section, please also add it to the gallery, both to `docs/gallery.py` and to `docs/gallery.rst`. (Make sure you have already installed Yellowbrick in editable mode, from the top level directory: `pip install -e` .)

If you want to regenerate a single image (e.g. the elbow curve plot), you can do so as follows:

```bash
$ python docs/gallery.py elbow
```

If you want to regenerate them all (note: this takes a long time!)

```bash
$ python docs/gallery.py all
```
