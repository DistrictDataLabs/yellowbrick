# yellowbrick.missing.base
# Base Visualizer for missing values
#
# Author:  Nathan Danielsen <nathan.danielsen@gmail.com>
# Created: Fri Mar 29 5:17:36 2018 -0500
#
# Copyright (C) 2018 District Data Labs
# For license information, see LICENSE.txt
#
# ID: base.py [] nathan.danielsen@gmail.com.com $

"""
Base classes for feature visualizers and feature selection tools.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from yellowbrick.features.base import DataVisualizer
from yellowbrick.utils import is_dataframe, is_structured_array
from sklearn.base import TransformerMixin


##########################################################################
## Feature Visualizers
##########################################################################

class MissingDataVisualizer(DataVisualizer):
    """
    """

    def __init__(self, ax=None, features=None, classes=None, color=None,
                 colormap=None, **kwargs):
        """
        Initialize the data visualization with many of the options required
        in order to make most visualizations work.
        """
        super(MissingDataVisualizer, self).__init__(ax=ax, features=features, **kwargs)
