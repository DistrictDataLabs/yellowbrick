# yellowbrick.features.pcoords
# Implementations of parallel coordinates for feature analysis.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Mon Oct 03 21:46:06 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: pcoords.py [] benjamin@bengfort.com $

"""
Implementations of parallel coordinates for multi-dimensional feature
analysis. There are a variety of parallel coordinates from Andrews Curves to
coordinates that optimize column order.
"""

##########################################################################
## Imports
##########################################################################

import matplotlib.pyplot as plt

from yellowbrick.base import FeatureVisualizer
from yellowbrick.exceptions import YellowbrickTypeError



##########################################################################
## Static Parallel Coordinates Visualizer
##########################################################################

class ParallelCoordinates(FeatureVisualizer):
    """
    Parallel coordinates displays each feature as a vertical axis spaced
    evenly along the horizontal, and each instance as a line drawn between
    each individual axis.
    """

    def __init__(self)
