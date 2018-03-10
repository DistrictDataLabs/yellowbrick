# yellowbrick.colors
# Colors and color helpers brought in from a different library.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Jun 24 17:02:53 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: colors.py [c6aff34] benjamin@bengfort.com $

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
        # so users could have a custom rcParams w/ no color...
        try:
            return [x['color'] for x in cyl]
        except KeyError:
            pass  # just return axes.color style below
    return mpl.rcParams['axes.color_cycle']


def resolve_colors(n_colors=None, colormap=None, colors=None):
    """
    Generates a list of colors based on common color arguments, for example
    the name of a colormap or palette or another iterable of colors. The list
    is then truncated (or multiplied) to the specific number of requested
    colors.

    Parameters
    ----------
    n_colors : int, default: None
        Specify the length of the list of returned colors, which will either
        truncate or multiple the colors available. If None the length of the
        colors will not be modified.

    colormap : str, default: None
        The name of the matplotlib color map with which to generate colors.

    colors : iterable, default: None
        A collection of colors to use specifically with the plot.

    Returns
    -------
    colors : list
        A list of colors that can be used in matplotlib plots.

    Notes
    -----
    This function was originally based on a similar function in the pandas
    plotting library that has been removed in the new version of the library.
    """

    # Work with the colormap if specified and colors is not
    if colormap is not None and colors is None:
        if isinstance(colormap, string_types):
            try:
                colormap = cm.get_cmap(colormap)
            except ValueError as e:
                raise YellowbrickValueError(e)


        n_colors = n_colors or len(get_color_cycle())
        _colors = list(map(colormap, np.linspace(0, 1, num=n_colors)))

    # Work with the color list
    elif colors is not None:

        # Warn if both colormap and colors is specified.
        if colormap is not None:
            warnings.warn(
                "both colormap and colors specified; using colors"
            )

        _colors = list(colors) # Ensure colors is a list

    # Get the default colors
    else:
        _colors = get_color_cycle()

    # Truncate or multiple the color list according to the number of colors
    if n_colors is not None and len(_colors) != n_colors:
        _colors = [
            _colors[idx % len(_colors)] for idx in np.arange(n_colors)
        ]

    return _colors


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
            # Must import here to avoid recursive import
            from .palettes import PALETTES

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
