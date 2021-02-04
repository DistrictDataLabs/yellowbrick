# yellowbrick.cluster.base
# Base class for cluster visualizers.
#
# Author:   Benjamin Bengfort
# Created:  Thu Mar 23 17:28:38 2017 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: base.py [241edca] benjamin@bengfort.com $

"""
Base class for cluster visualizers.
"""

##########################################################################
## Imports
##########################################################################

from yellowbrick.utils import isclusterer
from yellowbrick.base import ScoreVisualizer
from yellowbrick.exceptions import YellowbrickTypeError


## Packages for export
__all__ = ["ClusteringScoreVisualizer"]


##########################################################################
## Clustering Score Visualization Base Object
##########################################################################


class ClusteringScoreVisualizer(ScoreVisualizer):
    """
    Base class for all ScoreVisualizers that evaluate a clustering estimator.

    The primary functionality of this class is to perform a check to ensure
    that the wrapped estimator is a cluster estimator, otherwise a
    ``YellowbrickTypeError`` exception is raised.
    """

    def __init__(self, estimator, ax=None, fig=None, force_model=False, **kwargs):
        if not force_model and not isclusterer(estimator):
            raise YellowbrickTypeError(
                "The supplied model is not a clustering estimator; try a "
                "classifier or regression score visualizer instead!"
            )
        self.force_model = force_model
        super(ClusteringScoreVisualizer, self).__init__(
            estimator, ax=ax, fig=fig, **kwargs
        )
