# yellowbrick.color_utils
# Defines functions related to colors and palettes.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  Wed May 18 12:41:35 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: color_utils.py [15f72bf] pvomelveny@gmail.com $

"""
Defines color definitions and color maps specific to DDL and Yellowbrick.
"""

##########################################################################
## Imports
##########################################################################

from __future__ import print_function, division

import warnings
import numpy as np

import matplotlib.colors as mplcol
import matplotlib as mpl
import matplotlib.cm as cm

from yellowbrick.exceptions import YellowbrickValueError

# Check to see if matplotlib is at least sorta up to date
from distutils.version import LooseVersion
mpl_ge_150 = LooseVersion(mpl.__version__) >= "1.5.0"


##########################################################################
## Color Utils
##########################################################################


def get_color_cycle():
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
        try:
            colors = [c['color'] for c in list(plt.rcParams['axes.prop_cycle'])]
        except KeyError:
            colors = list(plt.rcParams.get('axes.color_cycle', list('bgrcmyk')))

        if isinstance(colors, str):
            colors = list(colors)

    if len(colors) != num_colors:
        multiple = num_colors // len(colors) - 1
        mod = num_colors % len(colors)
        colors += multiple * colors
        colors += colors[:mod]

    return colors
