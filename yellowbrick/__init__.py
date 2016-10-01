# yellowbrick
# A suite of visual analysis and diagnostic tools for machine learning.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  Wed May 18 10:46:33 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: __init__.py [0c5ba04] benjamin@bengfort.com $

"""
A suite of visual analysis and diagnostic tools to facilitate feature
selection, model selection, and parameter tuning for machine learning.
"""

##########################################################################
## Imports
##########################################################################
# Capture the original matplotlib rcParams
import matplotlib as mpl
_orig_rc_params = mpl.rcParams.copy()

from .version import get_version
from .anscombe import anscombe
# from .classifier import crplot, rocplot
from .regressor import peplot, residuals_plot
from .yb_rcmod import *
from .yb_palettes import *

##########################################################################
## Set default aesthetics
##########################################################################

set()

##########################################################################
## Package Version
##########################################################################

__version__ = get_version()
