# yellowbrick.color
# Defines color definitions and color maps specific to DDL and Yellowbrick.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  Wed May 18 12:41:35 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: color.py [] benjamin@bengfort.com $

"""
Defines color definitions and color maps specific to DDL and Yellowbrick.
"""

##########################################################################
## Imports
##########################################################################

from matplotlib import colors
from matplotlib.colors import ListedColormap

##########################################################################
## Colors
##########################################################################

ddl_heat = ['#DBDBDB','#DCD5CC','#DCCEBE','#DDC8AF','#DEC2A0','#DEBB91',\
            '#DFB583','#DFAE74','#E0A865','#E1A256','#E19B48','#E29539']

ddlheatmap = colors.ListedColormap(ddl_heat)
