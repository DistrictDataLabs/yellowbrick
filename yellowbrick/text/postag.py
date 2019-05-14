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
        If set to True, stacked barplot will be plotted.
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
        if(self.stack == True and y == None):
            raise YellowbrickValueError("Specify y for stack=True")
        
        labeled = []
        if(y!=None):
            for i in range(len(y)):
                labeled.append((X[0][i],y[i]))
        self.label_ = set(map(lambda x:x[1], labeled))
        
        if(self.stack == True):
            X = [[[y[0]] for y in labeled if y[1]==x] for x in self.label_]
        else:
            X = [X]
        
        self.last = None
        self.stack_count = 0
        
        for x in X:
            # TODO: add support for other tagsets?
            if self.tagset == "penn_treebank":
                self.pos_tag_counts_ = self._penn_tag_map()
                self._handle_treebank(x)
    
            elif self.tagset == "universal":
                self.pos_tag_counts_ = self._uni_tag_map()
                self._handle_universal(x)
            self.draw()

        return self

    def _penn_tag_map(self):
        """
        Returns a Penn Treebank part-of-speech tag map.
        """
        return {
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

    def _uni_tag_map(self):
        """
        Returns a Universal Dependencies part-of-speech tag map.
        """
        return {
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

        for tagged_doc in X:
            for tagged_sent in tagged_doc:
                for _, tag in tagged_sent:
                    if tag == "SPACE":
                        continue
                    self.pos_tag_counts_[jump.get(tag, "other")] += 1

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
        if(self.last == None):
            if self.stack:
                #Plot first layer of stack
                colors = resolve_colors(
                            n_colors=len(self.label_), 
                            colormap=self.colormap,
                            colors=self.colors
                            )
                self.ax.bar(
                    range(len(self.pos_tag_counts_)), 
                    list(self.pos_tag_counts_.values()),
                    color=colors[self.stack_count] 
                    )
                self.stack_count += 1
                self.last = list(self.pos_tag_counts_.values())
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
        else:
            #Plot stack barplot
            self.ax.bar(
                range(len(self.pos_tag_counts_)), 
                list(self.pos_tag_counts_.values()), 
                bottom=self.last, 
                color=colors[self.stack_count]
                )
            self.stack_count += 1
            self.last = [sum(x) for x in zip(list(self.last), 
                             list(self.pos_tag_counts_.values()))]
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
                (sum(self.pos_tag_counts_.values()))
            )
        )

        self.ax.set_xticks(range(len(self.pos_tag_counts_)))
        self.ax.set_xticklabels(list(self.pos_tag_counts_.keys()), rotation=90)

        # Set the axis labels
        if self.frequency:
            self.ax.set_xlabel(
                "{} part-of-speech tags, sorted by frequency".format(self.tagset_names[self.tagset])
            )
        else:
            self.ax.set_xlabel(
                "{} part-of-speech tags".format(self.tagset_names[self.tagset])
            )
        self.ax.set_ylabel("Count")
        if(self.stack == True):
            colors = resolve_colors(
                    n_colors=len(self.label_), 
                    colormap=self.colormap,
                    colors=self.colors
                    )
            manual_legend(self, self.label_, colors, loc='best')


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
