# yellowbrick.text.correlation
# Implementation of word correlation for text visualization.
#
# Author:   Patrick Deziel
# Created:  ---
#
# Copyright (C) 2022 The scikit-yb developers
# For license information, see LICENSE.txt
#

"""
Implementation of word correlation for text visualization.
"""


##########################################################################
## Imports
##########################################################################

from yellowbrick.text.base import TextVisualizer


##########################################################################
## Word Correlation Plot Visualizer
##########################################################################


class WordCorrelationPlot(TextVisualizer):
    """
    Word correlation illustrates the extent to which words in a corpus appear with each
    other.

    WordCorrelationPlot visualizes the binary correlation between words across
    documents as a heatmap. The correlation is defined using the phi-coefficient or
    mean square contingency coefficient between any two words m and n.

    Parameters
    ----------


    Attributes
    ----------

    """
    def __init__(self, ax=None, **kwargs):
        super(WordCorrelationPlot, self).__init__(ax, **kwargs)

    def fit(self, X, y=None):
        self.draw(X)
        return self

    def draw(self, X):
        self.ax.plt(X)
        return self.ax

    def finalize(self):
        self.set_title("Word Correlation Plot")