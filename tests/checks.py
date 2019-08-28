# tests.checks
# Performs checking that visualizers adhere to Yellowbrick conventions.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Mon May 22 11:18:06 2017 -0700
#
# Copyright (C) 2017 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: checks.py [4131cb1] benjamin@bengfort.com $

"""
Performs checking that visualizers adhere to Yellowbrick conventions.
"""

##########################################################################
## Imports
##########################################################################

import sys

sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

from yellowbrick.base import ModelVisualizer, ScoreVisualizer
from yellowbrick.classifier.base import ClassificationScoreVisualizer
from yellowbrick.cluster.base import ClusteringScoreVisualizer
from yellowbrick.features.base import FeatureVisualizer, DataVisualizer
from yellowbrick.regressor.base import RegressionScoreVisualizer
from yellowbrick.text.base import TextVisualizer


##########################################################################
## Checking runable
##########################################################################


def check_visualizer(Visualizer):
    """
    Check if visualizer adheres to Yellowbrick conventions.

    This function runs an extensive test-suite for input validation, return
    values, exception handling, and more. Additional tests for scoring or
    tuning visualizers will be run if the Visualizer clss inherits from the
    corresponding object.
    """
    name = Visualizer.__name__
    for check in _yield_all_checks(name, Visualizer):
        check(name, Visualizer)


##########################################################################
## Generate the specific per-visualizer checking
##########################################################################


def _yield_all_checks(name, Visualizer):
    """
    Composes the checks required for the specific visualizer.
    """

    # Global Checks
    yield check_instantiation
    yield check_estimator_api

    # Visualizer Type Checks
    if issubclass(Visualizer, RegressionScoreVisualizer):
        for check in _yield_regressor_checks(name, Visualizer):
            yield check

    if issubclass(Visualizer, ClassificationScoreVisualizer):
        for check in _yield_classifier_checks(name, Visualizer):
            yield check

    if issubclass(Visualizer, ClusteringScoreVisualizer):
        for check in _yield_clustering_checks(name, Visualizer):
            yield check

    if issubclass(Visualizer, FeatureVisualizer):
        for check in _yield_feature_checks(name, Visualizer):
            yield check

    if issubclass(Visualizer, TextVisualizer):
        for check in _yield_text_checks(name, Visualizer):
            yield check

    # Other checks


def _yield_regressor_checks(name, Visualizer):
    """
    Checks for regressor visualizers
    """
    pass


def _yield_classifier_checks(name, Visualizer):
    """
    Checks for classifier visualizers
    """
    pass


def _yield_clustering_checks(name, Visualizer):
    """
    Checks for clustering visualizers
    """
    pass


def _yield_feature_checks(name, Visualizer):
    """
    Checks for feature visualizers
    """
    pass


def _yield_text_checks(name, Visualizer):
    """
    Checks for text visualizers
    """
    pass


##########################################################################
## Checking Functions
##########################################################################


def check_instantiation(name, Visualizer, args, kwargs):
    # assert that visualizers can be passed an axes object.
    ax = plt.gca()

    viz = Visualizer(*args, **kwargs)
    assert viz.ax == ax


def check_estimator_api(name, Visualizer):
    X = np.random.rand((5, 10))
    y = np.random.randint(0, 2, 10)

    # Ensure fit returns self.
    viz = Visualizer()
    self = viz.fit(X, y)
    assert viz == self


if __name__ == "__main__":
    import sys

    sys.path.append("..")

    from yellowbrick.classifier import *
    from yellowbrick.cluster import *
    from yellowbrick.features import *
    from yellowbrick.regressor import *
    from yellowbrick.text import *

    visualizers = [
        ClassBalance,
        ClassificationReport,
        ConfusionMatrix,
        ROCAUC,
        KElbowVisualizer,
        SilhouetteVisualizer,
        ScatterVisualizer,
        JointPlotVisualizer,
        Rank2D,
        RadViz,
        ParallelCoordinates,
        AlphaSelection,
        ManualAlphaSelection,
        PredictionError,
        ResidualsPlot,
        TSNEVisualizer,
        FreqDistVisualizer,
        PosTagVisualizer,
    ]

    for visualizer in visualizers:
        check_visualizer(visualizer)
