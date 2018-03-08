# yellowbrick.cluster.base
# Base class for cluster visualizers.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Mar 23 17:28:38 2017 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: base.py [241edca] benjamin@bengfort.com $

"""
Base class for cluster visualizers.
"""

##########################################################################
## Imports
##########################################################################

from ..utils import isclusterer
from ..base import ScoreVisualizer
from ..exceptions import YellowbrickTypeError


## Packages for export
__all__ = [
    "ClusteringScoreVisualizer",
]


##########################################################################
## Clustering Score Visualization Base Object
##########################################################################

class ClusteringScoreVisualizer(ScoreVisualizer):
    """
    Base class for all ScoreVisualizers that evaluate a clustering estimator.

    The primary functionality of this class is to perform a check to ensure
    that the wrapped estimator is a cluster estimator, otherwise a
    ``YewllowbrickTypeError`` exception is raised.
    """

    def __init__(self, model, ax=None, **kwargs):
        if not isclusterer(model):
            raise YellowbrickTypeError(
                "The supplied model is not a clustering estimator; try a "
                "classifier or regression score visualizer instead!"
            )

        super(ClusteringScoreVisualizer, self).__init__(model, ax=ax, **kwargs)
