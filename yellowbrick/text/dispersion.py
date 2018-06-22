# yellowbrick.text.dispersion
# Implementations of lexical dispersions for text visualization.
#
# Author:   Larry Gray
# Created:  2018-06-21 10:06
#
# Copyright (C) 2018 District Data Labs
# For license information, see LICENSE.txt
#
# ID: dispersion.py [67b2740] lwgray@gmail.com $

"""
Implementation of lexical dispersion for text visualization
"""


##########################################################################
## Imports
##########################################################################

from yellowbrick.text.base import TextVisualizer

##########################################################################
## Dispersion Plot Visualizer
##########################################################################

class DispersionPlotVisualizer(TextVisualizer):
    """
    DispersionPlotVisualizer allows for visualization of the lexical dispersion
    of words in a corpus.  Lexical dispersion is a measure of a word's
    homeogeneity across the parts of a corpus.  This plot notes the occurences
    of a word and how many words from the beginning it appears.
    
    Parameters
    ----------
    words : list
        A list of words whose dispersion will be examined within a corpus 
    
    ax : matplotlib axes, default: None
        The axes to plot the figure on.
    
    color : list or tuple of colors
        Specify color for bars
        
    kwargs : dict
        Pass any additional keyword arguments to the super class.
    
    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """
    
    def __init__(self, words, ax=None, color=None, ignore_case=False, **kwargs):
        super(DispersionPlotVisualizer, self).__init__(ax=ax, **kwargs)
        
        self.color = color
        self.words = words
        self.ignore_case = ignore_case
        
    def fit(self, X):
        """
        The fit method is the primary drawing input for the dispersion
        visualization. It requires the corpus as a list of words.
        
        Parameters
        ----------
        X : list
            A list of words in the order they appear in the corpus.
        """
            
        text = list(X)
        self.words.reverse()

        if self.ignore_case:
            words_to_comp = list(map(str.lower, self.words))
            text_to_comp = list(map(str.lower, text))
        else:
            words_to_comp = self.words
            text_to_comp = text

        points = [(x,y) for x in range(len(text_to_comp))
                        for y in range(len(words_to_comp))
                        if text_to_comp[x] == words_to_comp[y]]
        if points:
            self.x_, self.y_ = list(zip(*points))
        else:
            self.x_ = self.y_ = ()
        
        self.draw()
        return self
    
    def draw(self, **kwargs):
        """
        Called from the fit method, this method creates the canvas and
        draws the distribution plot on it.
        Parameters
        ----------
        kwargs: generic keyword arguments.
        """
        
        self.ax.scatter(self.x_, self.y_, marker='|', color=self.color)
        self.ax.set_yticks(list(range(len(self.words))))
        self.ax.set_yticklabels(self.words)
        
    def finalize(self, **kwargs):
        """
        The finalize method executes any subclass-specific axes
        finalization steps. The user calls poof & poof calls finalize.
        Parameters
        ----------
        kwargs: generic keyword arguments.
        """
        
        self.ax.set_ylim(-1, len(self.words))
        self.ax.set_title("Lexical Dispersion Plot")
        self.ax.set_xlabel("Word Offset")
        self.ax.grid(False)


##########################################################################
## Quick Method
##########################################################################

def dispersion(words, corpus, ax=None, color=None, ignore_case=False, **kwargs):
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

    kwargs : dict
        Pass any additional keyword arguments to the super class.
        
    Returns
    -------
    ax: matplotlib axes
        Returns the axes that the plot was drawn on
    """
    
    # Instantiate the visualizer
    visualizer = DispersionPlotVisualizer(
        words, ax=ax, color=color, ignore_case=ignore_case, **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(corpus)

    # Return the axes object on the visualizer
    return visualizer.ax
