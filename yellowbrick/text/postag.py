# yellowbrick.text.postag
# Implementation of part-of-speech visualization for text.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  2017-03-05 14:44
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: postag.py [849f5a8] rebecca.bilbro@bytecubed.com $

"""
Implementation of part-of-speech visualization for text,
enabling the user to visualize a single document or
small subset of documents.
"""

##########################################################################
# Imports
##########################################################################

from yellowbrick.text.base import TextVisualizer

##########################################################################
# PosTagVisualizer
##########################################################################

class PosTagVisualizer(TextVisualizer):
    """
    A part-of-speech tag visualizer colorizes text to enable
    the user to visualize the proportions of nouns, verbs, etc.
    and to use this information to make decisions about
    part-of-speech tagging, text normalization (e.g. stemming
    vs lemmatization) and vectorization.

    Parameters
    ----------
    kwargs : dict
        Pass any additional keyword arguments to the super class.
    cmap : dict
        ANSII colormap

    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """
    def __init__(self, ax=None, **kwargs):
        """
        Initializes the base frequency distributions with many
        of the options required in order to make this
        visualization work.
        """
        super(PosTagVisualizer, self).__init__(ax=ax, **kwargs)

        # TODO: hard-coding in the ANSII colormap for now.
        # Can we let the user reset the colors here?
        self.COLORS = {
            'white'      : "\033[0;37m{}\033[0m",
            'yellow'     : "\033[0;33m{}\033[0m",
            'green'      : "\033[0;32m{}\033[0m",
            'blue'       : "\033[0;34m{}\033[0m",
            'cyan'       : "\033[0;36m{}\033[0m",
            'red'        : "\033[0;31m{}\033[0m",
            'magenta'    : "\033[0;35m{}\033[0m",
            'black'      : "\033[0;30m{}\033[0m",
            'darkwhite'  : "\033[1;37m{}\033[0m",
            'darkyellow' : "\033[1;33m{}\033[0m",
            'darkgreen'  : "\033[1;32m{}\033[0m",
            'darkblue'   : "\033[1;34m{}\033[0m",
            'darkcyan'   : "\033[1;36m{}\033[0m",
            'darkred'    : "\033[1;31m{}\033[0m",
            'darkmagenta': "\033[1;35m{}\033[0m",
            'darkblack'  : "\033[1;30m{}\033[0m",
             None        : "\033[0;0m{}\033[0m"
        }

        self.TAGS = {
            'NN'   : 'green',
            'NNS'  : 'green',
            'NNP'  : 'green',
            'NNPS' : 'green',
            'VB'   : 'blue',
            'VBD'  : 'blue',
            'VBG'  : 'blue',
            'VBN'  : 'blue',
            'VBP'  : 'blue',
            'VBZ'  : 'blue',
            'JJ'   : 'red',
            'JJR'  : 'red',
            'JJS'  : 'red',
            'RB'   : 'cyan',
            'RBR'  : 'cyan',
            'RBS'  : 'cyan',
            'IN'   : 'darkwhite',
            'POS'  : 'darkyellow',
            'PRP$' : 'magenta',
            'DT'   : 'black',
            'CC'   : 'black',
            'CD'   : 'black',
            'WDT'  : 'black',
            'WP'   : 'black',
            'WP$'  : 'black',
            'WRB'  : 'black',
            'EX'   : 'yellow',
            'FW'   : 'yellow',
            'LS'   : 'yellow',
            'MD'   : 'yellow',
            'PDT'  : 'yellow',
            'RP'   : 'yellow',
            'SYM'  : 'yellow',
            'TO'   : 'yellow',
            'None' : 'off'
        }

    def colorize(self, token, color):
        """
        Colorize text

        Parameters
        ----------
        token : str
            A str representation of

        """
        return self.COLORS[color].format(token)

    def transform(self, tagged_tuples):
        """
        The transform method transforms the raw text input for the
        part-of-speech tagging visualization. It requires that
        documents be in the form of (tag, token) tuples.

        Parameters
        ----------
        tagged_token_tuples : list of tuples
            A list of (tag, token) tuples

        Text documents must be tokenized and tagged before passing to fit()
        """
        self.tagged = [
            (self.TAGS.get(tag),tok) for tok, tag in tagged_tuples
        ]
        #
        # print(' '.join((colorize(token, color) for color, token in self.tagged)))
        # print('\n')
