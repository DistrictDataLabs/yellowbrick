# yellowbrick.pipeline
# Implements a visual pipeline that subclasses Scikit-Learn pipelines.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Oct 07 21:41:06 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: pipeline.py [1efae1f] benjamin@bengfort.com $

"""
Implements a visual pipeline that subclasses Scikit-Learn pipelines.
"""

##########################################################################
## Imports
##########################################################################

from os import path
from .base import Visualizer
from .utils.helpers import slugify
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

    def poof(self, outdir=None, ext=".pdf", **kwargs):
        """
        A single entry point to rendering all visualizations in the visual
        pipeline. The rendering for the output depends on the backend context,
        but for path based renderings (e.g. saving to a file), specify a
        directory and extension to compse an outpath to save each
        visualization (file names will be based on the  named step).

        Parameters
        ----------
        outdir : path
            The directory to save visualizations to.

        ext : string, default = ".pdf"
            The extension of the file to save the visualization to.

        kwargs : dict
            Keyword arguments to pass to the ``poof()`` method of all steps.
        """
        for name, step in self.visual_steps.items():
            if outdir is not None:
                outpath = path.join(outdir, slugify(name) + ext)
            else:
                outpath = None

            step.poof(outpath=outpath, **kwargs)

    def fit_transform_poof(self, X, y=None, outpath=None, **kwargs):
        """
        Fit the model and transforms and then call poof.
        """
        self.fit_transform(X, y, **kwargs)
        self.poof(outpath, **kwargs)
