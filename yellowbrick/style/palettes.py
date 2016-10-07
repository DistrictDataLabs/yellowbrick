# yellowbrick.style.palettes
# Implements the variety of colors that yellowbrick allows access to by name.
#
# Author:   Patrick O'Melveny <pvomelveny@gmail.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Tue Oct 04 15:30:15 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: palettes.py [] pvomelveny@gmail.com $

"""
Implements the variety of colors that yellowbrick allows access to by name.
This code was originally based on Seaborn's rcmody.py but has since been
cleaned up to be Yellowbrick-specific and to dereference tools we don't use.

Note that these functions alter the matplotlib rc dictionary on the fly.
"""

##########################################################################
## Imports
##########################################################################

from __future__ import division
from itertools import cycle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol

from six import string_types
from six.moves import range

from .colors import get_color_cycle
from yellowbrick.exceptions import YellowbrickValueError

##########################################################################
## Exports
##########################################################################

__all__ = ["color_palette", "set_color_codes"]


##########################################################################
## Special, Named Colors
##########################################################################

YB_KEY = '#111111'  # The yellowbrick key (black) color is very dark grey


##########################################################################
## Color Palettes
## Note all 6/7 color palettes can be mapped to bgrmyck color codes
## via the `set_color_codes` function, make sure they are ordered!
##########################################################################

PALETTES = {
    # "name": ['blue', 'green', 'red', 'maroon', 'yellow', 'cyan']
    # The yellowbrick default palette
    "yellowbrick": ['#0272a2', '#9fc377', '#ca0b03', '#a50258', '#d7c703', '#88cada'],

    # The following are from ColorBrewer
    "accent": ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f'],
    "dark":   ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02'],
    "pastel": ['#cbd5e8', '#b3e2cd', '#fdcdac', '#f4cae4', '#fff2ae', '#e6f5c9'],
    "bold":   ['#377eb8', '#4daf4a', '#e41a1c', '#984ea3', '#ffff33', '#ff7f00'],
    "muted":  ['#80b1d3', '#8dd3c7', '#fb8072', '#bebada', '#ffffb3', '#fdb462'],

    # The reset colors back to the original mpl color codes
    "reset":  ['#0000ff', '#008000', '#ff0000', '#bf00bf', '#bfbf00', '#00bfbf', '#000000'],

    # Colorblind colors
    "colorblind":     ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9"],
    "sns_colorblind": ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9"],

    # The following are Seaborn colors
    "sns_deep":   ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"],
    "sns_muted":  ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"],
    "sns_pastel": ["#92C6FF", "#97F0AA", "#FF9F9A", "#D0BBFF", "#FFFEA3", "#B0E0E6"],
    "sns_bright": ["#003FFF", "#03ED3A", "#E8000B", "#8A2BE2", "#FFC400", "#00D7FF"],
    "sns_dark":   ["#001C7F", "#017517", "#8C0900", "#7600A1", "#B8860B", "#006374"],

    # Other palettes
    "flatui":   ["#34495e", "#2ecc71", "#e74c3c", "#9b59b6", "#f4d03f", "#3498db"],

    # Longer palettes that do not map to bgrmyck color space.
    "ddl_heat": ['#DBDBDB', '#DCD5CC', '#DCCEBE', '#DDC8AF', '#DEC2A0', '#DEBB91',
                 '#DFB583', '#DFAE74', '#E0A865', '#E1A256', '#E19B48', '#E29539'],

    "paired":   ["#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a", "#ffff99", "#b15928",
                 "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#a6cee3", "#1f78b4"],

    "set1":     ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33",
                 "#a65628", "#f781bf", "#999999"],
}

## Special, backward compatible color map.
ddlheatmap = mplcol.ListedColormap(PALETTES["ddl_heat"])


##########################################################################
## Palette Object
##########################################################################

class ColorPalette(list):
    """
    A wrapper for functionality surrounding a list of colors, including a
    context manager that allows the palette to be set with a with statement.
    """

    def __init__(self, name_or_list):
        """
        Can initialize the ColorPalette with either a name or a list.

        Parameters
        ----------

        name_or_list :
            specify a palette name or a list of RGB or Hex values

        """
        if isinstance(name_or_list, string_types):
            if name_or_list not in PALETTES:
                raise YellowbrickValueError(
                    "'{}' is not a recognized palette!".format(name_or_list)
                )

            name_or_list = PALETTES[name_or_list]

        super(ColorPalette, self).__init__(name_or_list)

    def __enter__(self):
        """
        Open the context and assign the pallete to the mpl.rcParams
        """
        from .rcmod import set_palette
        self._orig_palette = color_palette()
        set_palette(self)
        return self

    def __exit__(self, *args):
        """
        Close the context and restore the original palette
        """
        from .rcmod import set_palette
        set_palette(self._orig_palette)

    def as_hex(self):
        """
        Return a color palette with hex codes instead of RGB values.
        """
        hex = [mpl.colors.rgb2hex(rgb) for rgb in self]
        return ColorPalette(hex)

    def as_rgb(self):
        """
        Return a color palette with RGB values instead of hex codes.
        """
        rgb = [mpl.colors.colorConverter.to_rgb(hex) for hex in self]
        return ColorPalette(rgb)

    def plot(self, size=1):
        """
        Plot the values in the color palatte as a horizontal array.
        See Seaborn's palplot function for inspiration.

        Parameters
        ----------
        size : int
            scaling factor for size of the plot

        """
        n = len(self)
        fig, ax = plt.subplots(1, 1, figsize=(n * size, size))
        ax.imshow(np.arange(n).reshape(1,n),
                  cmap=mpl.colors.ListedColormap(list(self)),
                  interpolation="nearest", aspect="auto")
        ax.set_xticks(np.arange(n) - .5)
        ax.set_yticks([-.5, .5])
        ax.set_xticklabels([])
        ax.set_yticklabels([])


##########################################################################
## Palette Functions
##########################################################################

def color_palette(palette=None, n_colors=None):
    """
    Return a color palette object with color definition and handling.

    Calling this function with ``palette=None`` will return the current
    matplotlib color cycle.

    This function can also be used in a ``with`` statement to temporarily
    set the color cycle for a plot or set of plots.

    Parameters
    ----------

    palette : None or str or sequence
        Name of a palette or ``None`` to return the current palette. If a
        sequence the input colors are used but possibly cycled.

        Available palette names from :py:mod:`yellowbrick.colors.palettes` are:

        .. hlist::
            :columns: 3

            * :py:const:`accent`
            * :py:const:`dark`
            * :py:const:`paired`
            * :py:const:`pastel`
            * :py:const:`bold`
            * :py:const:`muted`
            * :py:const:`colorblind`
            * :py:const:`sns_colorblind`
            * :py:const:`sns_deep`
            * :py:const:`sns_muted`
            * :py:const:`sns_pastel`
            * :py:const:`sns_bright`
            * :py:const:`sns_dark`
            * :py:const:`flatui`

    n_colors : None or int
        Number of colors in the palette. If ``None``, the default will depend
        on how ``palette`` is specified. Named palettes default to 6 colors
        which allow the use of the names "bgrmyck", though others do have more
        or less colors; therefore reducing the size of the list can only be
        done by specifying this parameter. Asking for more colors than exist
        in the palette will cause it to cycle.

    Returns
    -------
    list(tuple)
        Returns a ColorPalette object, which behaves like a list, but can be
        used as a context manager and possesses functions to convert colors.

    .. seealso::

        :func:`.set_palette`
            Set the default color cycle for all plots.
        :func:`.set_color_codes`
            Reassign color codes like ``"b"``, ``"g"``, etc. to
            colors from one of the yellowbrick palettes.
        :func:`..colors.resolve_colors`
            Resolve a color map or listed sequence of colors.

    """
    if palette is None:
        palette = get_color_cycle()
        if n_colors is None:
            n_colors = len(palette)

    elif not isinstance(palette, string_types):
        palette = palette
        if n_colors is None:
            n_colors = len(palette)

    else:
        if palette.lower() not in PALETTES:
            raise YellowbrickValueError(
                "'{}' is not a recognized palette!".format(palette)
            )

        palette = PALETTES[palette.lower()]
        if n_colors is None:
            n_colors = len(palette)

    # Always return as many colors as we asked for
    pal_cycle = cycle(palette)
    palette = [next(pal_cycle) for _ in range(n_colors)]

    # Always return in RGB tuple format
    try:
        palette = map(mpl.colors.colorConverter.to_rgb, palette)
        palette = ColorPalette(palette)
    except ValueError:
        raise YellowbrickValueError(
            "Could not generate a palette for %s" % str(palette)
        )

    return palette


def set_color_codes(palette="accent"):
    """
    Change how matplotlib color shorthands are interpreted.

    Calling this will change how shorthand codes like "b" or "g"
    are interpreted by matplotlib in subsequent plots.

    Parameters
    ----------
    palette : str
        Named yellowbrick palette to use as the source of colors.

    See Also
    --------
    set_palette : Color codes can also be set through the function that
                  sets the matplotlib color cycle.
    """

    if palette not in PALETTES:
        raise YellowbrickValueError(
            "'{}' is not a recognized palette!".format(palette)
        )

    # Fetch the colors and adapt the length
    colors = PALETTES[palette]

    if len(colors) > 7:
        # Truncate colors that are longer than 7
        colors = colors[:7]

    elif len(colors) < 7:
        # Add the key (black) color to colors that are shorter than 7
        colors = colors + [YB_KEY]

    # Set the color codes on matplotlib
    for code, color in zip("bgrmyck", colors):
        rgb = mpl.colors.colorConverter.to_rgb(color)
        mpl.colors.colorConverter.colors[code] = rgb
        mpl.colors.colorConverter.cache[code] = rgb
