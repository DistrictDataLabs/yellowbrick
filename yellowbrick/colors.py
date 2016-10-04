# yellowbrick.colors
# Colors and color helpers brought in from a different library.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Jun 24 17:02:53 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: colors.py [] benjamin@bengfort.com $

"""
Colors and color helpers brought in from an alternate library.
See https://bl.ocks.org/mbostock/5577023
"""

##########################################################################
## Imports
##########################################################################

import random

from copy import copy
from cloudscope.exceptions import BadValue

##########################################################################
## Color Palettes
##########################################################################

FLATUI = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
PAIRED = [
    "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a", "#ffff99", "#b15928",
    "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#a6cee3", "#1f78b4",
]
SET1   = [
    "#e41a1c", "#377eb8", "#4daf4a",
    "#984ea3", "#ff7f00", "#ffff33",
    "#a65628", "#f781bf", "#999999"
]

PALETTES = {
    'flatui': FLATUI,
    'paired': PAIRED,
    'set1': SET1,
}

##########################################################################
## Color Utilities
##########################################################################

class ColorMap(object):
    """
    A helper for mapping categorical values to colors on demand.
    """

    def __init__(self, colors='flatui', shuffle=False):
        """
        Specify either a list of colors or one of the color names. If shuffle
        is True then the colors will be shuffled randomly.
        """
        self.mapping = {}
        self.colors = colors

        if shuffle:
            random.shuffle(self._colors)

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, value):
        """
        Converts color strings into a color listing.
        """
        if isinstance(value, basestring):
            if value not in PALETTES:
                raise BadValue("'{}' is not a registered color palette")
            self._colors = copy(PALETTES[value])
        elif isinstance(value, list):
            self._colors = value
        else:
            self._colors = list(value)

    def __call__(self, category):
        if category not in self.mapping:
            if self.colors:
                self.mapping[category] = self.colors.pop()
            else:
                raise ValueError(
                    "Not enough colors for this many categories!"
                )

        return self.mapping[category]
