# yellowbrick.pipeline
# Implements a visual pipeline that subclasses Scikit-Learn pipelines.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Oct 07 21:41:06 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: pipeline.py [] benjamin@bengfort.com $

"""
Implements a visual pipeline that subclasses Scikit-Learn pipelines.
"""

##########################################################################
## Imports
##########################################################################

from .base import Visualizer
from sklearn.pipeline import Pipeline


##########################################################################
## Visual Pipeline
##########################################################################

class VisualPipeline(Pipeline):
    """Pipeline of transforms and visualizers with a final estimator.

    Sequentially apply a list of transforms, visualizers, and a final
    estimator which may be evaluated by additional visualizers.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement fit and transform methods.
    The final estimator only needs to implement fit.

    Any step that implements draw or poof methods can be called sequentially
    directly from the VisualPipeline, allowing multiple visual diagnostics to
    be generated, displayed, and saved on demand.
    If draw or poof is not called, the visual pipeline should be equivalent to
    the simple pipeline to ensure no reduction in performance.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters. These steps
    can be visually diagnosed by visualizers at every point in the pipeline.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator. Any intermediate step can be a FeatureVisualizer and the
        last step can be a ScoreVisualizer.

    Attributes
    ----------
    named_steps : dict
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are step parameters.

    visual_steps : dict
        Read-only attribute to access any visualizer in he pipeline by user
        given name. Keys are step names and values are visualizer steps.

    """

    @property
    def visual_steps(self):
        return dict(
            step for step in self.steps
            if isinstance(step[1], Visualizer)
        )

    def draw(self, *args, **kwargs):
        """
        Calls draw on steps (including the final estimator) that has a draw
        method and passes the args and kwargs to that draw function.
        """
        for name, step in self.visual_steps.items():
            step.draw(*args, **kwargs)

    def poof(self, *args, **kwargs):
        """
        Calls poof on steps (including the final estimator) that has a poof
        method and passes the args and kwargs to that poof function.
        """
        for name, step in self.visual_steps.items():
            step.poof(*args, **kwargs)

    def fit_transform_poof(self, X, y=None, **kwargs):
        """
        Fit the model and transforms and then call poof.
        """
        self.fit_transform(X, y, **kwargs)
        self.poof(**kwargs)
