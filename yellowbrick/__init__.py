# yellowbrick
# A suite of visual analysis and diagnostic tools for machine learning.
#
# Author:   Rebecca Bilbro
# Author:   Benjamin Bengfort
# Created:  Wed May 18 10:46:33 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
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

# Import the version number at the top level
from .version import get_version, __version_info__

# Import the style management functions
from .style.rcmod import reset_defaults, reset_orig
from .style.rcmod import set_aesthetic, set_style, set_palette
from .style.palettes import color_palette, set_color_codes

# Import yellowbrick functionality to the top level
# TODO: review top-level functionality
from .anscombe import anscombe
from .datasaurus import datasaurus
from .classifier import ROCAUC, ClassBalance, ClassificationScoreVisualizer

# from .classifier import crplot, rocplot
# from .regressor import peplot, residuals_plot


##########################################################################
## Set default aesthetics
##########################################################################

set_aesthetic()  # NOTE: modifies mpl.rcParams


##########################################################################
## Package Version
##########################################################################

__version__ = get_version(short=True)
