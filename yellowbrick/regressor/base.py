# yellowbrick.regressor.base
# Base classes for regressor Visualizers.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Jun 03 10:30:36 2016 -0700
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: base.py [7d3f5e6] benjamin@bengfort.com $

"""
Base classes for regressor Visualizers.
"""

##########################################################################
## Imports
##########################################################################

from ..utils import isregressor
from ..base import ScoreVisualizer
from ..exceptions import YellowbrickTypeError


## Packages for export
__all__ = [
    "RegressionScoreVisualizer",
]


##########################################################################
## Regression Visualization Base Object
##########################################################################

class RegressionScoreVisualizer(ScoreVisualizer):
    """
    Base class for all ScoreVisualizers that evaluate a regression estimator.

    The primary functionality of this class is to perform a check to ensure
    the passed in estimator is a regressor, otherwise it raises a
    ``YellowbrickTypeError``.
    """

    def __init__(self, model, ax=None, **kwargs):
        if not isregressor(model):
            raise YellowbrickTypeError(
                "This estimator is not a regressor; try a classifier or "
                "clustering score visualizer instead!"
        )

        super(RegressionScoreVisualizer, self).__init__(model, ax=ax, **kwargs)
