# yellowbrick.text.postag
# Implementation of part-of-speech visualization for text.
#
# Author:   Rebecca Bilbro 
# Created:  2019-02-18 13:12
#
# Copyright (C) 2019 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: postag.py bilbro@gmail.com $

"""
Implementation of part-of-speech visualization for text,
enabling the user to visualize a single document or
small subset of documents.
"""

##########################################################################
# Imports
##########################################################################

from yellowbrick.text.base import TextVisualizer
from yellowbrick.style.colors import resolve_colors


##########################################################################
# Part-of-speech tag punctuation and labels
##########################################################################

PUNCT_LIST = [".", ":", ",", "``", "''", "(", ")", "#", "$"]
TAGSET_NAMES = {
    "penn_treebank" : "Penn Treebank",
    "universal" : "Universal Dependencies"
}


##########################################################################
# PosTagVisualizer
##########################################################################

class PosTagVisualizer(TextVisualizer):
    """
    A part-of-speech tag visualizer that creates a bar chart to visualize 
    the relative proportions of different parts-of-speech in a corpus
    
    PosTagVisualize requires documents be in the form of (tag, token) 
    tuples.

    Parameters
    ----------
    ax : matplotlib axes
        The axes to plot the figure on.
    tagset: string
        The tagset that was used to perform part-of-speech tagging.
        Either "penn_treebank" or "universal", defaults to "penn_treebank".
        Use "universal" if corpus has been tagged using SpaCy.
    colors : list or tuple of colors
        Specify the colors for each individual part-of-speech.
    colormap : string or matplotlib cmap
        Specify a colormap to color the parts-of-speech.
    kwargs : dict
        Pass any additional keyword arguments to the PosTagVisualizer.
    """
    def __init__(
        self, 
        ax=None, 
        tagset="penn_treebank", 
        colormap=None, 
        colors=None, 
        **kwargs
    ):
        super(PosTagVisualizer, self).__init__(ax=ax, **kwargs)
       
        self.tagset_names = TAGSET_NAMES
        
        if tagset not in self.tagset_names:
            raise NotImplementedError(
                "PosTagVisualizer not implemented for {} tags.".format(
                    tagset
                )
            )
        else:
            self.tagset = tagset
        
        self.punct_list = PUNCT_LIST
        self.colormap = colormap
        self.colors = colors

    def fit(self, X, y=None, **kwargs):
        """
        Fits the corpus to the appropriate tag map.
        Text documents must be tokenized & tagged before passing to fit

        Parameters
        ----------
        X : list or generator
            Should be provided as a list of documents or a generator
            that yields a list of documents that contain a list of 
            sentences that contain (token, tag) tuples.

        y : ndarray or Series of length n
            An optional array or series of target or class values for
            instances.

        kwargs : dict
            Pass generic arguments to the drawing method

        Returns
        -------
        self : instance
            Returns the instance of the transformer/visualizer
        """
        # TODO: add support for other tagsets?
        if self.tagset == "penn_treebank":
            self.tag_map = self._penn_tag_map()
            self._handle_treebank(X)

        elif self.tagset == "universal":
            self.tag_map = self._uni_tag_map()
            self._handle_universal(X)
      
        self.draw()
        return self
            
    def _penn_tag_map(self):
        """
        Returns a Penn Treebank part-of-speech tag map.
        """
        return {
            "noun" : 0,
            "verb" : 0,
            "adjective" : 0,
            "adverb" : 0,
            "preposition" : 0,
            "determiner" : 0,
            "pronoun" : 0,
            "conjunction" : 0,
            "infinitive" : 0,
            "wh- word" : 0,
            "modal" : 0,
            "possessive" : 0,  
            "existential" : 0,
            "punctuation" : 0,
            "digit" : 0,
            "non-English" : 0,
            "interjection" : 0,
            "list" : 0,
            "symbol": 0,
            "other" : 0
        }
    
    def _uni_tag_map(self):
        """
        Returns a Universal Dependencies part-of-speech tag map.
        """
        return {
            "noun" : 0,
            "verb" : 0,
            "adjective" : 0,
            "adverb" : 0,
            "adposition" : 0,
            "determiner" : 0,
            "pronoun" : 0,
            "conjunction" : 0,
            "infinitive" : 0,
            "punctuation" : 0,
            "number" : 0,
            "interjection" : 0,
            "symbol": 0,
            "other" : 0
        }

    def _handle_universal(self, X):
        """
        Scan through the corpus to compute counts of each Universal
        Dependencies part-of-speech.

        Parameters
        ----------
        X : list or generator
            Should be provided as a list of documents or a generator
            that yields a list of documents that contain a list of 
            sentences that contain (token, tag) tuples.
        """
        for tagged_doc in X:
            for tagged_sent in tagged_doc:
                for _, tag in tagged_sent:
                    # combine proper and regular nouns
                    if tag in ["NOUN", "PROPN"]:
                        self.tag_map["noun"] += 1
                    elif tag == "ADJ":
                        self.tag_map["adjective"] += 1
                    elif tag == "VERB":
                        self.tag_map["verb"] += 1 
                    # include particles with adverbs
                    elif tag in ["ADV", "PART"]:
                        self.tag_map["adverb"] += 1
                    elif tag == "ADP":
                        self.tag_map["adposition"] += 1
                    elif tag == "PRON":
                        self.tag_map["pronoun"] += 1
                    elif tag == "CCONJ":
                        self.tag_map["conjunction"] += 1
                    elif tag == "PUNCT":
                        self.tag_map["punctuation"] += 1
                    elif tag == "DET":
                        self.tag_map["determiner"] += 1
                    elif tag == "NUM":
                        self.tag_map["number"] += 1
                    elif tag == "INTJ":
                        self.tag_map["interjection"] += 1
                    elif tag == "SYM":
                        self.tag_map["symbol"] += 1
                    elif tag == "SPACE":
                        continue
                    else:
                        self.tag_map["other"] += 1                
                
    def _handle_treebank(self, X):
        """
        Create a part-of-speech tag mapping using the Penn Treebank tags
        
        Parameters
        ----------
        X : list or generator
            Should be provided as a list of documents or a generator
            that yields a list of documents that contain a list of 
            sentences that contain (token, tag) tuples.
        """
        for tagged_doc in X:
            for tagged_sent in tagged_doc:
                for _, tag in tagged_sent:
                    if tag.startswith("N"):
                        self.tag_map["noun"] += 1
                    elif tag.startswith("J"):
                        self.tag_map["adjective"] += 1
                    elif tag.startswith("V"):
                        self.tag_map["verb"] += 1
                    # include particles with adverbs
                    elif tag.startswith("RB") or tag == "RP":
                        self.tag_map["adverb"] += 1
                    elif tag.startswith("PR"):
                        self.tag_map["pronoun"] += 1
                    elif tag.startswith("W"):
                        self.tag_map["wh- word"] += 1
                    elif tag == "CC":
                        self.tag_map["conjunction"] += 1
                    elif tag == "CD":
                        self.tag_map["digit"] += 1
                    # combine predeterminer and determiner
                    elif tag in ["DT" or "PDT"]:
                        self.tag_map["determiner"] += 1
                    elif tag == "EX":
                        self.tag_map["existential"] += 1
                    elif tag == "FW":
                        self.tag_map["non-English"] += 1
                    elif tag == "IN":
                        self.tag_map["preposition"] += 1
                    elif tag == "POS":
                        self.tag_map["possessive"] += 1
                    elif tag == "LS":
                        self.tag_map["list"] += 1
                    elif tag == "MD":
                        self.tag_map["modal"] += 1
                    elif tag in self.punct_list:
                        self.tag_map["punctuation"] += 1
                    elif tag == "TO":
                        self.tag_map["infinitive"] += 1
                    elif tag == "UH":
                        self.tag_map["interjection"] += 1
                    elif tag == "SYM":
                        self.tag_map["symbol"] += 1
                    else:
                        self.tag_map["other"] += 1
                
    def draw(self, **kwargs):
        """
        Called from the fit method, this method creates the canvas and
        draws the part-of-speech tag mapping as a bar chart.

        Parameters
        ----------
        kwargs: dict
            generic keyword arguments.

        Returns
        -------
        ax : matplotlib axes
            Axes on which the PosTagVisualizer was drawn.
        """
        colors = resolve_colors(
            n_colors=len(self.tag_map), 
            colormap=self.colormap, 
            colors=self.colors
        )
        self.ax.bar(
            range(len(self.tag_map)), 
            list(self.tag_map.values()), 
            color=colors
        )
        return self.ax
        
    def finalize(self, **kwargs):
        """
        Finalize the plot with ticks, labels, and title

        Parameters
        ----------
        kwargs: dict
            generic keyword arguments.
        """
        # NOTE: not deduping here, so this is total, not unique
        self.set_title(
            "PosTag plot for {}-token corpus".format(
                (sum(self.tag_map.values()))
            )
        )
        
        self.ax.set_xticks(range(len(self.tag_map)))
        self.ax.set_xticklabels([self.tag_map.keys()], rotation=90)
        
        # Set the axis labels
        self.ax.set_xlabel(
            '{} part-of-speech tags'.format(
                self.tagset_names[self.tagset]
            )
        )
        self.ax.set_ylabel('Count')

##########################################################################
## Quick Method
##########################################################################

def postag(X, y=None, ax=None, tagset="penn_treebank", colormap=None, 
            colors=None, **kwargs):
    """
    Display a barchart with the counts of different parts of speech
    in X, which consists of a part-of-speech-tagged corpus, which the
    visualizer expects to be a list of lists of lists of (token, tag) 
    tuples.

    Parameters
    ----------
    ax : matplotlib axes
        The axes to plot the figure on.
    tagset: string
        The tagset that was used to perform part-of-speech tagging.
        Either "penn_treebank" or "universal", defaults to "penn_treebank".
        Use "universal" if corpus has been tagged using SpaCy.
    colors : list or tuple of colors
        Specify the colors for each individual part-of-speech.
    colormap : string or matplotlib cmap
        Specify a colormap to color the parts-of-speech.
    kwargs : dict
        Pass any additional keyword arguments to the PosTagVisualizer. 

    Returns
    -------
    ax : matplotlib axes
        Returns the axes on which the PosTagVisualizer was drawn.
    """
    # Instantiate the visualizer
    visualizer = PosTagVisualizer(
        ax, tagset, colors, colormap, **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y, **kwargs)

    # Return the axes object on the visualizer
    return visualizer.ax
        
        
        