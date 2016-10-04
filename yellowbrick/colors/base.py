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
from six import string_types
from yellowbrick.exceptions import YellowbrickValueError


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
        if isinstance(value, string_types):
            if value not in PALETTES:
                raise YellowbrickValueError(
                    "'{}' is not a registered color palette".format(value)
                )
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
                raise YellowbrickValueError(
                    "Not enough colors for this many categories!"
                )

        return self.mapping[category]
