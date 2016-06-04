# yellowbrick.yb_palettes
# Defines color definitions and color maps specific to DDL and Yellowbrick.
#
# Original based on Seaborn's rcmod.py:
# <https://github.com/mwaskom/seaborn/>
# For license information, see LICENSE.txt
#
# TODO: Clean up docs so they don't reference Seaborn things we don't have

"""Functions that alter the matplotlib rc dictionary on the fly."""

##########################################################################
## Imports
##########################################################################
from __future__ import division
from itertools import cycle

import matplotlib as mpl
from six import string_types
from six.moves import range

from .color_utils import get_color_cycle


##########################################################################
## Exports
##########################################################################
__all__ = ["color_palette", "set_color_codes"]

##########################################################################
## Default Yellowbrick Palettes (and Default Seaborn, just cause)
##########################################################################
# Taken from Colorbrewer, qualitative color schemes
YELLOWBRICK_PALETTES = dict(
    accent=['#7fc97f', '#beaed4', '#fdc086',
            '#ffff99', '#386cb0', '#f0027f'],
    dark=['#1b9e77', '#d95f02', '#7570b3',
          '#e7298a', '#66a61e', '#e6ab02'],
    paired=['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c',
            '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00',
            '#cab2d6', '#6a3d9a'],
    pastel=['#b3e2cd', '#fdcdac', '#cbd5e8',
             '#f4cae4', '#e6f5c9', '#fff2ae'],
    bold=['#e41a1c', '#377eb8', '#4daf4a',
          '#984ea3', '#ff7f00', '#ffff33'],
    muted=['#8dd3c7', '#ffffb3', '#bebada',
           '#fb8072', '#80b1d3', '#fdb462']
)

SEABORN_PALETTES = dict(
    sns_deep=["#4C72B0", "#55A868", "#C44E52",
          "#8172B2", "#CCB974", "#64B5CD"],
    sns_muted=["#4878CF", "#6ACC65", "#D65F5F",
           "#B47CC7", "#C4AD66", "#77BEDB"],
    sns_pastel=["#92C6FF", "#97F0AA", "#FF9F9A",
            "#D0BBFF", "#FFFEA3", "#B0E0E6"],
    sns_bright=["#003FFF", "#03ED3A", "#E8000B",
            "#8A2BE2", "#FFC400", "#00D7FF"],
    sns_dark=["#001C7F", "#017517", "#8C0900",
          "#7600A1", "#B8860B", "#006374"],
    sns_colorblind=["#0072B2", "#009E73", "#D55E00",
                "#CC79A7", "#F0E442", "#56B4E9"]
    )
##########################################################################
## Palette Functions
##########################################################################
class _ColorPalette(list):
    """Set the color palette in a with statement, otherwise be a list."""
    def __enter__(self):
        """Open the context."""
        from .yb_rcmod import set_palette
        self._orig_palette = color_palette()
        set_palette(self)
        return self

    def __exit__(self, *args):
        """Close the context."""
        from .yb_rcmod import set_palette
        set_palette(self._orig_palette)

    def as_hex(self):
        """Return a color palette with hex codes instead of RGB values."""
        hex = [mpl.colors.rgb2hex(rgb) for rgb in self]
        return _ColorPalette(hex)


def color_palette(palette=None, n_colors=None, desat=None):
    """Return a list of colors defining a color palette.
    Availible seaborn palette names:
        accent, dark, paired, pastel, bold, muted
    Availible seaborn palette names:
        sns_deep, sns_muted, sns_bright, sns_pastel, sns_dark, sns_colorblind
    Other options:
        list of colors
    Calling this function with ``palette=None`` will return the current
    matplotlib color cycle.
    This function can also be used in a ``with`` statement to temporarily
    set the color cycle for a plot or set of plots.
    Parameters
    ----------
    palette: None, string, or sequence, optional
        Name of palette or None to return current palette. If a sequence, input
        colors are used but possibly cycled and desaturated.
    n_colors : int, optional
        Number of colors in the palette. If ``None``, the default will depend
        on how ``palette`` is specified. Named palettes default to 6 colors
        (except paired, which has 10),
        but grabbing the current palette or passing in a list of colors will
        not change the number of colors unless this is specified. Asking for
        more colors than exist in the palette will cause it to cycle.

    Returns
    -------
    palette : list of RGB tuples.
        Color palette. Behaves like a list, but can be used as a context
        manager and possesses an ``as_hex`` method to convert to hex color
        codes.
    See Also
    --------
    set_palette : Set the default color cycle for all plots.
    set_color_codes : Reassign color codes like ``"b"``, ``"g"``, etc. to
                      colors from one of the yellowbrick palettes.
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
        if palette.lower() == "jet":
            raise ValueError("No.")
        elif palette in YELLOWBRICK_PALETTES:
            palette = YELLOWBRICK_PALETTES[palette]
        elif palette in SEABORN_PALETTES:
            palette = SEABORN_PALETTES[palette]
        else:
            raise ValueError("%s is not a valid palette "
                             "name in yellowbrick" % palette)
        if n_colors is None:
            n_colors = len(palette)

    # Always return as many colors as we asked for
    pal_cycle = cycle(palette)
    palette = [next(pal_cycle) for _ in range(n_colors)]

    # Always return in r, g, b tuple format
    try:
        palette = map(mpl.colors.colorConverter.to_rgb, palette)
        palette = _ColorPalette(palette)
    except ValueError:
        raise ValueError("Could not generate a palette for %s" % str(palette))

    return palette

def set_color_codes(palette="accent"):
    """Change how matplotlib color shorthands are interpreted.
    Calling this will change how shorthand codes like "b" or "g"
    are interpreted by matplotlib in subsequent plots.
    Parameters
    ----------
    palette : {accent, dark, paired, pastel, bold, muted}
        Named yellowbrick palette to use as the source of colors.
    See Also
    --------
    set_palette : Color codes can also be set through the function that
                  sets the matplotlib color cycle.
    """
    if palette == "reset":
        colors = [(0., 0., 1.), (0., .5, 0.), (1., 0., 0.), (.75, .75, 0.),
                  (.75, .75, 0.), (0., .75, .75), (0., 0., 0.)]
    else:
        colors = YELLOWBRICK_PALETTES[palette] + [(.1, .1, .1)]
    for code, color in zip("bgrmyck", colors):
        rgb = mpl.colors.colorConverter.to_rgb(color)
        mpl.colors.colorConverter.colors[code] = rgb
        mpl.colors.colorConverter.cache[code] = rgb
