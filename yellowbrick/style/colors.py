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
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

from copy import copy
from six import string_types
from yellowbrick.exceptions import YellowbrickValueError

# Check to see if matplotlib is at least sorta up to date
from distutils.version import LooseVersion
mpl_ge_150 = LooseVersion(mpl.__version__) >= "1.5.0"


##########################################################################
## Color Utilities
##########################################################################

def get_color_cycle():
    """
    Returns the current color cycle from matplotlib.
    """
    if mpl_ge_150:
        cyl = mpl.rcParams['axes.prop_cycle']
        # matplotlib 1.5 verifies that axes.prop_cycle *is* a cycler
        # but no garuantee that there's a `color` key.
        # so users could have a custom rcParmas w/ no color...
        try:
            return [x['color'] for x in cyl]
        except KeyError:
            pass  # just return axes.color style below
    return mpl.rcParams['axes.color_cycle']


def resolve_colors(num_colors=None, colormap=None, color=None):
    """
    Resolves the colormap or the color list with the number of colors.
    See: https://github.com/pydata/pandas/blob/master/pandas/tools/plotting.py#L163

    Parameters
    ----------
    num_colors : int or None
        the number of colors in the cycle or colormap

    colormap : str or None
        the colormap used to create the sequence of colors

    color : list or Non e
        the list of colors to specifically use with the plot 

    """

    # Work with the colormap
    if color is None and colormap is None:
        if isinstance(colormap, str):
            cmap = colormap
            colormap = cm.get_cmap(colormap)

            if colormap is None:
                raise YellowbrickValueError(
                    "Colormap {0} is not a valid matploblib cmap".format(cmap)
                )

        colors = list(map(colormap, np.linspace(0, 1, num=num_colors)))

    # Work with the color list
    elif color is not None:

        if colormap is not None:
            warnings.warn(
                "'color' and 'colormap' cannot be used simultaneously! Using 'color'."
            )

        colors = list(color) # Ensure colors is a list

    # Get the default colors
    else:
        colors = get_color_cycle()

    if len(colors) != num_colors:
        multiple = num_colors // len(colors) - 1
        mod = num_colors % len(colors)
        colors += multiple * colors
        colors += colors[:mod]

    return colors


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
