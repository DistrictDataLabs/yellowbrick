# yellowbrick.text.dispersion
# Implementations of lexical dispersions for text visualization.
#
# Author:   Larry Gray
# Created:  Fri Jun 22 15:40:49 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: dispersion.py [3822dd6] lwgray@gmail.com $

"""
Implementation of lexical dispersion for text visualization
"""


##########################################################################
## Imports
##########################################################################

import itertools
from collections import defaultdict

import numpy as np

from yellowbrick.text.base import TextVisualizer
from yellowbrick.style.colors import resolve_colors
from yellowbrick.exceptions import YellowbrickValueError


##########################################################################
## Dispersion Plot Visualizer
##########################################################################


class DispersionPlot(TextVisualizer):
    """
    Lexical dispersion illustrates the homogeneity of a word (or set of words) across
    the documents of a corpus.

    DispersionPlot allows for visualization of the lexical dispersion of words in a
    corpus. This plot illustrates with vertical lines the occurrences of one or more
    search terms throughout the corpus, noting how many words relative to the beginning
    of the corpus it appears. If the target vector of the corpus documents is provided,
    the points will be colored with respect to their document category, which allows for
    additional analysis of relationships in search term homogeneity within and across
    document categories. If annotation is requested, document boundaries will be
    displayed as vertical lines in the plot.

    Parameters
    ----------
    search_terms : list
        A list of search terms whose dispersion across a corpus passed at fit
        should be visualized.

    ax : matplotlib axes, default: None
        The axes to plot the figure on.

    colors : list or tuple of colors
        Specify the colors for each individual class. Will override colormap if both are
        provided.

    colormap : string or matplotlib cmap
        Qualitative colormap for discrete target

    ignore_case : boolean, default: False
        Specify whether input will be case-sensitive.

    annotate_docs : boolean, default: False
        Specify whether document boundaries will be displayed. Vertical lines
        are positioned at the end of each document.

    labels : list of strings
        The names of the classes in the target, used to create a legend.
        Labels must match names of classes in sorted order.

    kwargs : dict
        Pass any additional keyword arguments to the super class.

    Attributes
    ----------
    self.classes_ : list
        A list of strings representing the unique classes in the target in sorted order.
        If ``y`` is provided, these are extracted from ``y``, unless a list of class
        labels is provided by the user on instantiation.

    self.boundaries_ : list
        A list of integers indicating the document boundaries with respect to
        word offsets.

    self.indexed_words_ : list
        A list of integers indicating the y position for each occurrence of each of
        the search terms.

    self.word_categories_ : list
        A list of strings indicating the corresponding document category of each search
        term occurrence.
    """

    # NOTE: cannot be np.nan
    NULL_CLASS = None

    def __init__(
        self,
        search_terms,
        ax=None,
        colors=None,
        colormap=None,
        ignore_case=False,
        annotate_docs=False,
        labels=None,
        **kwargs
    ):
        super(DispersionPlot, self).__init__(ax=ax, **kwargs)

        self.labels = labels
        self.colors = colors
        self.colormap = colormap

        self.ignore_case = ignore_case
        self.search_terms = search_terms
        self.annotate_docs = annotate_docs

    def _compute_dispersion(self, X, y):
        """
        Produces a generator containing the offset word count, y_coordinate, and
        label/category for each occurrance of the search terms.

        Attributes
        ----------
        self.boundaries_ : list
            A list of integers indicating the document boundaries with respect to
            word offsets.
        """
        self.boundaries_ = []
        offset = 0

        if y is None:
            y = itertools.repeat(None)

        for doc, category in zip(X, y):
            for word in doc:
                if self.ignore_case:
                    word = word.lower()

                # NOTE: this will find all indices if duplicate words are supplied
                # In the case that word is not in target words, any empty list is
                # returned and no data will be yielded
                offset += 1
                for y_coord in (self.indexed_words_ == word).nonzero()[0]:
                    y_coord = int(y_coord)
                    yield (offset, y_coord, category)

            if self.annotate_docs:
                self.boundaries_.append(offset)

        self.boundaries_ = np.array(self.boundaries_, dtype=int)

    def _check_missing_words(self, points):
        """
        Helper method to raise an error if any of the requested search
        terms do not appear in the corpus.
        """
        for index in range(len(self.indexed_words_)):
            if index in points[:, 1]:
                pass
            else:
                raise YellowbrickValueError(
                    ("The search term '{}' is not found in " "this corpus").format(
                        self.indexed_words_[index]
                    )
                )

    def fit(self, X, y=None, **kwargs):
        """
        The fit method is the primary drawing input for the dispersion
        visualization.

        Parameters
        ----------
        X : list or generator
            Should be provided as a list of documents or a generator
            that yields a list of documents that contain a list of
            words in the order they appear in the document.

        y : ndarray or Series of length n
            An optional array or series of target or class values for
            instances. If this is specified, then the points will be colored
            according to their class.

        Returns
        -------
        self : instance
            Returns the instance of the transformer/visualizer

        Attributes
        ----------
        self.classes_ : list
            A list of strings representing the unique classes in the target in sorted order.
            If ``y`` is provided, these are extracted from ``y``, unless a list of class
            labels is provided by the user on instantiation.

        self.indexed_words_ : list
            A list of integers indicating the y position for each occurrence of each of
            the search terms.

        self.word_categories_ : list
            A list of strings indicating the corresponding document category of each search
            term occurrence.
        """

        if y is not None:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = np.array([self.NULL_CLASS])

        # Create an index (e.g. the y position) for the target words
        self.indexed_words_ = np.flip(self.search_terms, axis=0)
        if self.ignore_case:
            self.indexed_words_ = np.array([w.lower() for w in self.indexed_words_])

        # Stack is used to create a 2D array from the generator
        try:
            offsets_positions_categories = np.stack(self._compute_dispersion(X, y))
        except ValueError:
            raise YellowbrickValueError(("No search terms were found in the corpus"))

        word_positions = np.stack(
            zip(
                offsets_positions_categories[:, 0].astype(int),
                offsets_positions_categories[:, 1].astype(int),
            )
        )

        self.word_categories_ = offsets_positions_categories[:, 2]

        self._check_missing_words(word_positions)

        self.draw(word_positions, **kwargs)
        return self

    def draw(self, points, **kwargs):
        """
        Called from the fit method, this method creates the canvas and
        draws the plot on it.

        Parameters
        ----------
        kwargs: generic keyword arguments.
        """

        # Resolve the labels with the classes
        labels = self.labels if self.labels is not None else self.classes_
        if len(labels) != len(self.classes_):
            raise YellowbrickValueError(
                (
                    "number of supplied labels ({}) does not "
                    "match the number of classes ({})"
                ).format(len(labels), len(self.classes_))
            )

        # Create the color mapping for the labels.
        color_values = resolve_colors(
            n_colors=len(labels), colormap=self.colormap, colors=self.colors
        )
        colors = dict(zip(labels, color_values))

        # Transform labels into a map of class to label
        labels = dict(zip(self.classes_, labels))

        # Define boundaries with a vertical line
        if self.annotate_docs:
            for xcoords in self.boundaries_:
                self.ax.axvline(x=xcoords, color="lightgray", linestyle="dashed")

        series = defaultdict(lambda: {"x": [], "y": []})

        if self.word_categories_ is not None:
            for point, t in zip(points, self.word_categories_):
                label = labels[t]
                series[label]["x"].append(point[0])
                series[label]["y"].append(point[1])
        else:
            label = self.classes_[0]
            for x, y in points:
                series[label]["x"].append(x)
                series[label]["y"].append(y)

        for label, points in series.items():
            self.ax.scatter(
                points["x"],
                points["y"],
                marker="|",
                c=colors[label],
                zorder=100,
                label=label,
            )

        self.ax.set_yticks(list(range(len(self.indexed_words_))))
        self.ax.set_yticklabels(self.indexed_words_)

        return self.ax

    def finalize(self, **kwargs):
        """
        Prepares the figure for rendering by adding a title, axis labels, and
        managing the limits of the text labels. Adds a legend outside of the plot.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        Notes
        -----
        Generally this method is called from show and not directly by the user.
        """
        self.set_title("Lexical Dispersion Plot")
        self.ax.set_ylim(-1, len(self.indexed_words_))
        self.ax.set_xlabel("Word Offset")
        self.ax.grid(False)

        # Add the legend outside of the figure box.
        if not all(self.classes_ == np.array([self.NULL_CLASS])):
            box = self.ax.get_position()
            self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            self.ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))


##########################################################################
## Quick Method
##########################################################################


def dispersion(
    search_terms,
    corpus,
    y=None,
    ax=None,
    colors=None,
    colormap=None,
    annotate_docs=False,
    ignore_case=False,
    labels=None,
    show=True,
    **kwargs
):
    """ Displays lexical dispersion plot for words in a corpus

    This helper function is a quick wrapper to utilize the DispersionPlot
    Visualizer for one-off analysis

    Parameters
    ----------

    search_terms : list
        A list of words whose dispersion will be examined within a corpus

    corpus : list
        Should be provided as a list of documents that contain
        a list of words in the order they appear in the document.

    y : ndarray or Series of length n
        An optional array or series of target or class values for
        instances. If this is specified, then the points will be colored
        according to their class.

    ax : matplotlib axes, default: None
        The axes to plot the figure on.

    colors : list or tuple of colors
        Specify the colors for each individual class. Will override colormap if both are
        provided.

    colormap : string or matplotlib cmap
        Qualitative colormap for discrete target

    annotate_docs : boolean, default: False
        Specify whether document boundaries will be displayed.  Vertical lines
        are positioned at the end of each document.

    ignore_case : boolean, default: False
        Specify whether input  will be case-sensitive.

    labels : list of strings
        The names of the classes in the target, used to create a legend.
        Labels must match names of classes in sorted order.

    show : bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however
        you cannot call ``plt.savefig`` from this signature, nor
        ``clear_figure``. If False, simply calls ``finalize()``

    kwargs : dict
        Pass any additional keyword arguments to the super class.

    Returns
    -------
    viz: DispersionPlot
        Returns the fitted, finalized visualizer
    """

    # Instantiate the visualizer
    visualizer = DispersionPlot(
        search_terms,
        ax=ax,
        colors=colors,
        colormap=colormap,
        ignore_case=ignore_case,
        labels=labels,
        annotate_docs=annotate_docs,
        **kwargs
    )

    visualizer.fit(corpus, y, **kwargs)

    if show:
        visualizer.show()
    else:
        visualizer.finalize()

    return visualizer
