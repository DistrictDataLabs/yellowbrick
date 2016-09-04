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
import matplotlib.colors as mplcol
import matplotlib as mpl


# Check to see if matplotlib is at least sorta up to date
from distutils.version import LooseVersion
mpl_ge_150 = LooseVersion(mpl.__version__) >= "1.5.0"

# TODO: This block should probably be moved/removed soon
##########################################################################
## Compatability with old stuff for now
##########################################################################

ddl_heat = ['#DBDBDB', '#DCD5CC', '#DCCEBE', '#DDC8AF', '#DEC2A0', '#DEBB91',
            '#DFB583','#DFAE74', '#E0A865', '#E1A256', '#E19B48', '#E29539']

ddlheatmap = mplcol.ListedColormap(ddl_heat)

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
