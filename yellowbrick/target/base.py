# yellowbrick.target.base
# Base classes for target visualizers
#
# Author:  Benjamin Bengfort
# Created: Thu Jul 19 09:25:53 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: base.py [d742c57] benjamin@bengfort.com $

"""
Base classes for target visualizers
"""

##########################################################################
# Imports
##########################################################################

from yellowbrick.base import Visualizer


##########################################################################
# TargetVisualizer Base Class
##########################################################################


class TargetVisualizer(Visualizer):
    """
    The base class for target visualizers, generic enough to support any
    computation on a single vector, y. This Visualizer is based on the
    LabelEncoder in sklearn.preprocessing, which only accepts a target y.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    fig : matplotlib Figure, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

    kwargs : dict
        Keyword arguments that are passed to the base class
    """

    def fit(self, y):
        """
        Fit the visualizer to the target y. Note that this visualizer breaks
        the standard estimator interface, and therefore cannot be used inside
        of pipelines, but must be used separately; similar to how the
        LabelEncoder is used.
        """
        raise NotImplementedError("target visualizers must implement a fit method")
