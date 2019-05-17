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
from yellowbrick.draw import manual_legend
from yellowbrick.exceptions import YellowbrickValueError

import  numpy as np
##########################################################################
# Part-of-speech tag punctuation and labels
##########################################################################

# NOTE: Penn Treebank converts all sentence closers (!,?,;) to periods
PUNCT_TAGS = [".", ":", ",", "``", "''", "(", ")", "#", "$"]
TAGSET_NAMES = {
    "penn_treebank": "Penn Treebank",
    "universal": "Universal Dependencies"
}


##########################################################################
# PosTagVisualizer
##########################################################################

class PosTagVisualizer(TextVisualizer):
    """
    Parts of speech (e.g. verbs, nouns, prepositions, adjectives)
    indicate how a word is functioning within the context of a sentence.
    In English as in many other languages, a single word can function in
    multiple ways. Part-of-speech tagging lets us encode information not
    only about a word’s definition, but also its use in context (for
    example the words “ship” and “shop” can be either a verb or a noun,
    depending on the context).

    The PosTagVisualizer creates a bar chart to visualize the relative
    proportions of different parts-of-speech in a corpus.

    Note that the PosTagVisualizer requires documents to already be
    part-of-speech tagged; the visualizer expects the corpus to come in
    the form of a list of (document) lists of (sentence) lists of
    (tag, token) tuples.

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
    frequency: bool {True, False}, default: False
        If set to True, part-of-speech tags will be plotted according to frequency,
        from most to least frequent.
    stack : bool {True, False}, default : False
        Plot the PosTag frequency chart as a per-class stacked bar chart. 
        Note that fit() requires y for this visualization.
    kwargs : dict
        Pass any additional keyword arguments to the PosTagVisualizer.

    Attributes
    ----------
    pos_tag_counts_: dict
        Mapping of part-of-speech tags to counts.

    Examples
    --------
    >>> viz = PosTagVisualizer()
    >>> viz.fit(X)
    >>> viz.poof()
    """

    def __init__(
        self,
        ax=None,
        tagset="penn_treebank",
        colormap=None,
        colors=None,
        frequency=False,
        stack=False,
        **kwargs
    ):
        super(PosTagVisualizer, self).__init__(ax=ax, **kwargs)

        self.tagset_names = TAGSET_NAMES

        if tagset not in self.tagset_names:
            raise YellowbrickValueError((
                "'{}' is an invalid tagset. Please choose one of {}."
                ).format(
                    tagset, ", ".join(self.tagset_names.keys())
            ))
        else:
            self.tagset = tagset

        self.punct_tags = frozenset(PUNCT_TAGS)
        self.colormap = colormap
        self.colors = colors
        self.frequency = frequency
        self.stack=stack

    def fit(self, X, y=None, **kwargs):
        """
        Fits the corpus to the appropriate tag map.
        Text documents must be tokenized & tagged before passing to fit.

        Parameters
        ----------
        X : list or generator
            Should be provided as a list of documents or a generator
            that yields a list of documents that contain a list of
            sentences that contain (token, tag) tuples.

        y : ndarray or Series of length n
            An optional array of target values that are ignored by the
            visualizer.

        kwargs : dict
            Pass generic arguments to the drawing method

        Returns
        -------
        self : instance
            Returns the instance of the transformer/visualizer
        """
        if self.stack and y is None:
            raise YellowbrickValueError("Specify y for stack=True")
        
        self.labels_=np.unique(y)
        
        if self.tagset == "penn_treebank":
            self.pos_tag_counts_ = self._penn_tag_map()
            self._handle_treebank(X,y)
        elif self.tagset == "universal":
            self.pos_tag_counts_ = self._uni_tag_map()
            self._handle_universal(X,y)
        self.draw()
        return self

    def _penn_tag_map(self):
        """
        Returns a Penn Treebank part-of-speech tag map.
        """
        
        default_dic =  {
                            "noun": 0,
                            "verb": 0,
                            "adjective": 0,
                            "adverb": 0,
                            "preposition": 0,
                            "determiner": 0,
                            "pronoun": 0,
                            "conjunction": 0,
                            "infinitive": 0,
                            "wh- word": 0,
                            "modal": 0,
                            "possessive": 0,
                            "existential": 0,
                            "punctuation": 0,
                            "digit": 0,
                            "non-English": 0,
                            "interjection": 0,
                            "list": 0,
                            "symbol": 0,
                            "other": 0,
                        }
        if self.stack:
            penn_tag_map={}
            for i in self.labels_:
                penn_tag_map[i]=default_dic.copy()
            return penn_tag_map
        else:
            return default_dic

    def _uni_tag_map(self):
        """
        Returns a Universal Dependencies part-of-speech tag map.
        """
        default_dic = {
                        "noun": 0,
                        "verb": 0,
                        "adjective": 0,
                        "adverb": 0,
                        "adposition": 0,
                        "determiner": 0,
                        "pronoun": 0,
                        "conjunction": 0,
                        "infinitive": 0,
                        "punctuation": 0,
                        "number": 0,
                        "interjection": 0,
                        "symbol": 0,
                        "other": 0,
                    }
        if self.stack:
            uni_tag_map={}
            for i in self.labels_:
                uni_tag_map[i]=default_dic.copy()
            return uni_tag_map
        else:
            return default_dic

    def _handle_universal(self, X,y):
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
            jump = {
                # combine proper and regular nouns
                "NOUN": "noun", "PROPN": "noun",
                "ADJ": "adjective",
                "VERB": "verb",
                # include particles with adverbs
                "ADV": "adverb", "PART": "adverb",
                "ADP": "adposition",
                "PRON": "pronoun",
                "CCONJ": "conjunction",
                "PUNCT": "punctuation",
                "DET": "determiner",
                "NUM": "number",
                "INTJ": "interjection",
                "SYM": "symbol",
            }
            if self.stack: 
                for tagged_doc in X:
                    for tagged_sent, y in zip(tagged_doc, y):
                        for _, tag in tagged_sent:
                            if tag == "SPACE":
                                continue
                            self.pos_tag_counts_[y][jump.get(tag, "other")] += 1
            else:
                for tagged_doc in X:
                    for tagged_sent in tagged_doc:
                        for _, tag in tagged_sent:
                            if tag == "SPACE":
                                continue
                            self.pos_tag_counts_[jump.get(tag, "other")] += 1

    def _handle_treebank(self, X,y):
        """
        Create a part-of-speech tag mapping using the Penn Treebank tags

        Parameters
        ----------
        X : list or generator
            Should be provided as a list of documents or a generator
            that yields a list of documents that contain a list of
            sentences that contain (token, tag) tuples.
        """
        if self.stack:
            for tagged_doc in X:
                for tagged_sent,y in zip(tagged_doc,y):
                    for _, tag in tagged_sent:
                        if tag.startswith("N"):
                            self.pos_tag_counts_[y]["noun"] += 1
                        elif tag.startswith("J"):
                            self.pos_tag_counts_[y]["adjective"] += 1
                        elif tag.startswith("V"):
                            self.pos_tag_counts_[y]["verb"] += 1
                        # include particles with adverbs
                        elif tag.startswith("RB") or tag == "RP":
                            self.pos_tag_counts_[y]["adverb"] += 1
                        elif tag.startswith("PR"):
                            self.pos_tag_counts_[y]["pronoun"] += 1
                        elif tag.startswith("W"):
                            self.pos_tag_counts_[y]["wh- word"] += 1
                        elif tag == "CC":
                            self.pos_tag_counts_[y]["conjunction"] += 1
                        elif tag == "CD":
                            self.pos_tag_counts_[y]["digit"] += 1
                        # combine predeterminer and determiner
                        elif tag in ["DT" or "PDT"]:
                            self.pos_tag_counts_[y]["determiner"] += 1
                        elif tag == "EX":
                            self.pos_tag_counts_[y]["existential"] += 1
                        elif tag == "FW":
                            self.pos_tag_counts_[y]["non-English"] += 1
                        elif tag == "IN":
                            self.pos_tag_counts_[y]["preposition"] += 1
                        elif tag == "POS":
                            self.pos_tag_counts_[y]["possessive"] += 1
                        elif tag == "LS":
                            self.pos_tag_counts_[y]["list"] += 1
                        elif tag == "MD":
                            self.pos_tag_counts_[y]["modal"] += 1
                        elif tag in self.punct_tags:
                            self.pos_tag_counts_[y]["punctuation"] += 1
                        elif tag == "TO":
                            self.pos_tag_counts_[y]["infinitive"] += 1
                        elif tag == "UH":
                            self.pos_tag_counts_[y]["interjection"] += 1
                        elif tag == "SYM":
                            self.pos_tag_counts_[y]["symbol"] += 1
                        else:
                            self.pos_tag_counts_[y]["other"] += 1
        else:
            for tagged_doc in X:
                for tagged_sent in tagged_doc:
                    for _, tag in tagged_sent:
                        if tag.startswith("N"):
                            self.pos_tag_counts_["noun"] += 1
                        elif tag.startswith("J"):
                            self.pos_tag_counts_["adjective"] += 1
                        elif tag.startswith("V"):
                            self.pos_tag_counts_["verb"] += 1
                        # include particles with adverbs
                        elif tag.startswith("RB") or tag == "RP":
                            self.pos_tag_counts_["adverb"] += 1
                        elif tag.startswith("PR"):
                            self.pos_tag_counts_["pronoun"] += 1
                        elif tag.startswith("W"):
                            self.pos_tag_counts_["wh- word"] += 1
                        elif tag == "CC":
                            self.pos_tag_counts_["conjunction"] += 1
                        elif tag == "CD":
                            self.pos_tag_counts_["digit"] += 1
                        # combine predeterminer and determiner
                        elif tag in ["DT" or "PDT"]:
                            self.pos_tag_counts_["determiner"] += 1
                        elif tag == "EX":
                            self.pos_tag_counts_["existential"] += 1
                        elif tag == "FW":
                            self.pos_tag_counts_["non-English"] += 1
                        elif tag == "IN":
                            self.pos_tag_counts_["preposition"] += 1
                        elif tag == "POS":
                            self.pos_tag_counts_["possessive"] += 1
                        elif tag == "LS":
                            self.pos_tag_counts_["list"] += 1
                        elif tag == "MD":
                            self.pos_tag_counts_["modal"] += 1
                        elif tag in self.punct_tags:
                            self.pos_tag_counts_["punctuation"] += 1
                        elif tag == "TO":
                            self.pos_tag_counts_["infinitive"] += 1
                        elif tag == "UH":
                            self.pos_tag_counts_["interjection"] += 1
                        elif tag == "SYM":
                            self.pos_tag_counts_["symbol"] += 1
                        else:
                            self.pos_tag_counts_["other"] += 1

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
        if self.stack:
            #Plot stack barchart
            colors = resolve_colors(
                        n_colors=len(self.labels_), 
                        colormap=self.colormap,
                        colors=self.colors
                        )
            last=[0]*len(self.pos_tag_counts_[self.labels_[0]])
            for i in range(len(self.labels_)):
                self.ax.bar(
                range(len(self.pos_tag_counts_[self.labels_[i]])), 
                list(self.pos_tag_counts_[self.labels_[i]].values()),
                bottom=last,
                color=colors[i] 
                )
                last = [sum(x) for x in zip(list(last), 
                         list(self.pos_tag_counts_[self.labels_[i]].values()))]
        else:
            # Plot barplot(not stacked)
            colors = resolve_colors(
                n_colors=len(self.pos_tag_counts_), 
                colormap=self.colormap, 
                colors=self.colors
                )
            if self.frequency:
                # Sort tags with respect to frequency in corpus
                sorted_tags = sorted(
                    self.pos_tag_counts_, key=self.pos_tag_counts_.get, 
                    reverse=True
                    )
                sorted_counts = [self.pos_tag_counts_[tag] for tag in sorted_tags]
    
                self.ax.bar(range(len(sorted_tags)), sorted_counts, 
                            color=colors)
                return self.ax
    
            self.ax.bar(
                range(len(self.pos_tag_counts_)),
                list(self.pos_tag_counts_.values()),
                color=colors,
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
        
        self.ax.set_ylabel("Count")
        if self.stack:
            self.ax.set_xticks(range(len(self.pos_tag_counts_[self.labels_[0]])))
            self.ax.set_xticklabels(list(self.pos_tag_counts_[self.labels_[0]].keys()),
                                    rotation=90)
            colors = resolve_colors(
                    n_colors=len(self.labels_), 
                    colormap=self.colormap,
                    colors=self.colors
                    )
            manual_legend(self, self.labels_, colors, loc='best')
            self.set_title(
                "PosTag plot for {}-token corpus".format(
                    (sum([sum(i.values()) for i in self.pos_tag_counts_.values()]))
                )
            )
        else:
            self.ax.set_xticks(range(len(self.pos_tag_counts_)))
            self.ax.set_xticklabels(list(self.pos_tag_counts_.keys()), rotation=90)
            if self.frequency:
                self.ax.set_xlabel(
                    "{} part-of-speech tags, sorted by frequency".format(
                            self.tagset_names[self.tagset])
                )
            else:
                self.ax.set_xlabel(
                    "{} part-of-speech tags".format(self.tagset_names[self.tagset])
                )
            self.set_title(
                "PosTag plot for {}-token corpus".format(
                    (sum(self.pos_tag_counts_.values()))
                )
            )


    def poof(self, outpath=None, **kwargs):
        if outpath is not None:
            kwargs["bbox_inches"] = kwargs.get("bbox_inches", "tight")
        return super(PosTagVisualizer, self).poof(outpath, **kwargs)


##########################################################################
## Quick Method
##########################################################################

def postag(
    X,
    y=None,
    ax=None,
    tagset="penn_treebank",
    colormap=None,
    colors=None,
    frequency=False,
    stack=False,
    **kwargs
):

    """
    Display a barchart with the counts of different parts of speech
    in X, which consists of a part-of-speech-tagged corpus, which the
    visualizer expects to be a list of lists of lists of (token, tag)
    tuples.

    Parameters
    ----------
    X : list or generator
        Should be provided as a list of documents or a generator
        that yields a list of documents that contain a list of
        sentences that contain (token, tag) tuples.
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
    frequency: bool {True, False}, default: False
        If set to True, part-of-speech tags will be plotted according to frequency,
        from most to least frequent.
    kwargs : dict
        Pass any additional keyword arguments to the PosTagVisualizer.

    Returns
    -------
    ax : matplotlib axes
        Returns the axes on which the PosTagVisualizer was drawn.
    """
    # Instantiate the visualizer
    visualizer = PosTagVisualizer(
        ax=ax, tagset=tagset, colors=colors, colormap=colormap,
        frequency=frequency, stack=stack, **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y=y, **kwargs)

    # Return the axes object on the visualizer
    return visualizer

