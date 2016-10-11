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

# Import the version number at the top level
from .version import get_version

# Import the style management functions
from .style.rcmod import *
from .style.palettes import *

# Import yellowbrick functionality to the top level
from .anscombe import anscombe
# from .classifier import crplot, rocplot
# from .regressor import peplot, residuals_plot

##########################################################################
## Set default aesthetics
##########################################################################

set_aesthetic() # modifies mpl.rcParams

##########################################################################
## Package Version
##########################################################################

__version__ = get_version(short=True)
