# yellowbrick.text.dispersion
# Implementations of lexical dispersions for text visualization.
#
# Author:   Larry Gray
# Created:  2018-06-21 10:06
#
# Copyright (C) 2018 District Data Labs
# For license information, see LICENSE.txt
#
# ID: dispersion.py [] lwgray@gmail.com $

"""
Implementation of lexical dispersion for text visualization
"""


##########################################################################
## Imports
##########################################################################

from yellowbrick.text.base import TextVisualizer
import numpy as np

##########################################################################
## Dispersion Plot Visualizer
##########################################################################

class DispersionPlot(TextVisualizer):
    """
    DispersionPlotVisualizer allows for visualization of the lexical dispersion
    of words in a corpus.  Lexical dispersion is a measure of a word's
    homeogeneity across the parts of a corpus.  This plot notes the occurences
    of a word and how many words from the beginning it appears.

    Parameters
    ----------
    words : list
        A list of target words whose dispersion across a corpus passed at fit
	will be visualized.

    ax : matplotlib axes, default: None
        The axes to plot the figure on.

    color : list or tuple of colors
        Specify color for bars

    ignore_case : boolean, default: False
	Specify whether input  will be case-sensitive.

    annotate_docs : boolean, default: False
        Specify whether document boundaries will be displayed.  Vertical lines
        are positioned at the end of each document.

    kwargs : dict
        Pass any additional keyword arguments to the super class.

    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """

    def __init__(self, words, ax=None, color=None, ignore_case=False,
                 annotate_docs=False, **kwargs):
        super(DispersionPlot, self).__init__(ax=ax, **kwargs)

        self.color = color
        self.words = words
        self.ignore_case = ignore_case
        self.annotate_docs = annotate_docs

    def _compute_dispersion(self, text):
        self.boundaries_ = []
        self.offset = 0
        for doc in text:
            for word in doc:
                if self.ignore_case:
                    word = word.lower()

                # NOTE: this will find all indices if duplicate words are supplied
                # In the case that word is not in target words, any empty list is
                # returned and no data will be yielded
                self.offset += 1
                for y in (self.target_words_ == word).nonzero()[0]:
                    yield (self.offset, y)
            if self.annotate_docs:
                self.boundaries_.append(self.offset)
        self.boundaries_ = np.array(self.boundaries_, dtype=int)

    def fit(self, text):
        """
        The fit method is the primary drawing input for the dispersion
        visualization. It requires the corpus as a list of words.

        Parameters
        ----------
        text : list
            Should be provided as a list of documents that contain
            a list of words in the order they appear in the document.
        """

        # Create an index (e.g. the y position) for the target words
        self.target_words_ = np.flip(self.words, axis=0)
        if self.ignore_case:
            self.target_words_ = np.array([w.lower() for w in self.target_words_])

        # Stack is used to create a 2D array from the generator
        points = np.stack(self._compute_dispersion(text))
        self.draw(points)
        return self

    def draw(self, points, **kwargs):
        """
        Called from the fit method, this method creates the canvas and
        draws the distribution plot on it.
        Parameters
        ----------
        kwargs: generic keyword arguments.
        """

        # Define boundaries with a vertical line
        if self.annotate_docs:
            for xcoords in self.boundaries_:
                self.ax.axvline(x=xcoords, color='lightgray', linestyle='dashed')

        self.ax.scatter(points[:,0], points[:,1], marker='|', color=self.color,
                        zorder=100)
        self.ax.set_yticks(list(range(len(self.target_words_))))
        self.ax.set_yticklabels(self.target_words_)

    def finalize(self, **kwargs):
        """
        The finalize method executes any subclass-specific axes
        finalization steps. The user calls poof & poof calls finalize.
        Parameters
        ----------
        kwargs: generic keyword arguments.
        """

        self.ax.set_ylim(-1, len(self.target_words_))
        self.ax.set_title("Lexical Dispersion Plot")
        self.ax.set_xlabel("Word Offset")
        self.ax.grid(False)


##########################################################################
## Quick Method
##########################################################################

def dispersion(words, corpus, ax=None, color=None,
               annotate_docs=False, ignore_case=False, **kwargs):
    """ Displays lexical dispersion plot for words in a corpus

    This helper function is a quick wrapper to utilize the DisperstionPlot
    Visualizer for one-off analysis

    Parameters
    ----------

    words : list
        A list of words whose dispersion will be examined within a corpus

    corpus : list
        A list of words in the order they appear in the corpus

    ax : matplotlib axes, default: None
        The axes to plot the figure on.

    color : list or tuple of colors
        Specify color for bars

    annotate_docs : boolean, default: False
        Specify whether document boundaries will be displayed.  Vertical lines
        are positioned at the end of each document.

    ignore_case : boolean, default: False
	Specify whether input  will be case-sensitive.

    kwargs : dict
        Pass any additional keyword arguments to the super class.

    Returns
    -------
    ax: matplotlib axes
        Returns the axes that the plot was drawn on
    """

    # Instantiate the visualizer
    visualizer = DispersionPlot(
        words, ax=ax, color=color, ignore_case=ignore_case,
        annotate_docs=annotate_docs, **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(corpus)

    # Return the axes object on the visualizer
    return visualizer.ax
