# yellowbrick.target.base
# Base classes for target visualizers
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Thu Jul 19 09:25:53 2018 -0400
#
# ID: base.py [] benjamin@bengfort.com $

"""
Base classes for target visualizers
"""

##########################################################################
## Imports
##########################################################################

from ..base import Visualizer


##########################################################################
## TargetVisualizer Base Class
##########################################################################

class TargetVisualizer(Visualizer):
    """
    The base class for target visualizers, generic enough to support any
    computation on a single vector, y. This Visualizer is based on the
    LabelEncoder in sklearn.preprocessing, which only accepts a target y.
    """

    def fit(self, y):
        """
        Fit the visualizer to the target y. Note that this visualizer breaks
        the standard estimator interface, and therefore cannot be used inside
        of pipelines, but must be used separately; similar to how the
        LabelEncoder is used.
        """
        raise NotImplementedError(
            "target visualizers must implement a fit method"
        )
