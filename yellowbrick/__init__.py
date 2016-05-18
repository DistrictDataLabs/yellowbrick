# yellowbrick
# A suite of visual analysis and diagnostic tools for machine learning.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  Wed May 18 10:46:33 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: __init__.py [] benjamin@bengfort.com $

"""
A suite of visual analysis and diagnostic tools to facilitate feature
selection, model selection, and parameter tuning for machine learning.
"""

##########################################################################
## Imports
##########################################################################

from .version import get_version
from .anscombe import anscombe
from .classifier import crplot, rocplot_compare

##########################################################################
## Package Version
##########################################################################

__version__ = get_version()
