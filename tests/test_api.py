# tests.test_api
# Ensures that standard visualizers adhere to the Yellowbrick API.
#
# Author:   Benjamin Bengfort
# Created:  Mon May 22 11:18:06 2017 -0700
#
# Copyright (C) 2017 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: checks.py [4131cb1] benjamin@bengfort.com $

"""
Ensures that standard visualizers adhere to the Yellowbrick API.

This module runs a full suite of checks against all of our documented Visualizers to
ensure they conform to our API. Visualizers that are considered "complete" should be
added to this test suite to ensure that they meet the requirements of the checks.
"""

##########################################################################
## Imports
##########################################################################

import pytest
import unittest.mock as mock
import matplotlib.pyplot as plt

from yellowbrick.base import *
from yellowbrick.pipeline import *
from yellowbrick.classifier import *
from yellowbrick.cluster import *
from yellowbrick.features import *
from yellowbrick.gridsearch import *
from yellowbrick.regressor import *
from yellowbrick.text import *
from yellowbrick.target import *
from yellowbrick.model_selection import *

from tests.fixtures import Dataset, Split
from yellowbrick.datasets import load_hobbies

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Lasso, LassoCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split as tts
from sklearn.datasets import make_blobs, make_classification, make_regression


BASES = [
    Visualizer,                      # green
    ModelVisualizer,                 # green
    ScoreVisualizer,                 # yellow
    ClassificationScoreVisualizer,   # red -> green
    RegressionScoreVisualizer,       # red -> yellow (needs tests)
    ClusteringScoreVisualizer,       # green
    TargetVisualizer,                # yellow (needs tests)
    FeatureVisualizer,               # green
    MultiFeatureVisualizer,          # green
    DataVisualizer,                  # green
    RankDBase,                       # green
    ProjectionVisualizer,            # green
    TextVisualizer,                  # green
    GridSearchVisualizer,            # red (prototype, no tests)
]

OTHER = [
    Wrapper,                         # green
    VisualizerGrid,                  # red (prototype)
    VisualPipeline,                  # red (prototype)
]

CLASSIFICATION_VISUALZERS = [
    ClassPredictionError,            # yellow
    ClassificationReport,            # green
    ConfusionMatrix,                 # green
    PrecisionRecallCurve,            # green
    ROCAUC,                          # green
    DiscriminationThreshold,         # green
]

CLUSTERING_VISUALIZERS = [
    KElbowVisualizer,                # yellow (problems with kneed)
    InterclusterDistance,            # yellow
    SilhouetteVisualizer,            # yellow --> green (quick method)
]

FEATURE_VISUALIZERS = [
    ExplainedVariance,               # red (undocumented, no tests, still a prototype)
    FeatureImportances,              # yellow --> green (getting moved)
    Rank1D,                          # green
    Rank2D,                          # green
    RFECV,                           # yellow --> green (getting moved)
    JointPlot,                       # red
    Manifold,                        # green
    PCA,                             # yellow (needs much better documentation)
    ParallelCoordinates,             # green
    RadialVisualizer,                # green
]

MODEL_SELECTION_VISUALIZERS = [
    CVScores,                        # green (style)
    LearningCurve,                   # green
    ValidationCurve,                 # green
    GridSearchColorPlot,             # red (prototype, untested, undocumented)
]

REGRESSOR_VISUALIZERS = [
    AlphaSelection,                  # red -> yellow (quick method)
    ManualAlphaSelection,            # red
    CooksDistance,                   # green
    PredictionError,                 # green
    ResidualsPlot,                   # green
]

TARGET_VISUALIZERS = [
    BalancedBinningReference,        # yellow (style)
    ClassBalance,                    # green
    FeatureCorrelation,              # green (its fine)
]

TEXT_VISUALIZERS = [
    DispersionPlot,                  # green
    FreqDistVisualizer,              # yellow (needs better test coverage)
    PosTagVisualizer,                # green
    TSNEVisualizer,                  # green
    UMAPVisualizer,                  # green
]

VISUALIZERS = (
    CLASSIFICATION_VISUALZERS
    + CLUSTERING_VISUALIZERS
    + FEATURE_VISUALIZERS
    + MODEL_SELECTION_VISUALIZERS
    + REGRESSOR_VISUALIZERS
    + TARGET_VISUALIZERS
    + TEXT_VISUALIZERS
)

QUICK_METHODS = {
    ClassPredictionError: class_prediction_error,
    ClassificationReport: classification_report,
    ConfusionMatrix: confusion_matrix,
    PrecisionRecallCurve: precision_recall_curve,
    ROCAUC: roc_auc,
    DiscriminationThreshold: discrimination_threshold,
    KElbowVisualizer: kelbow_visualizer,
    InterclusterDistance: intercluster_distance,
    SilhouetteVisualizer: silhouette_visualizer,
    ExplainedVariance: explained_variance_visualizer,
    FeatureImportances: feature_importances,
    Rank1D: rank1d,
    Rank2D: rank2d,
    RFECV: rfecv,
    JointPlot: joint_plot,                       # raises not implemented error
    Manifold: manifold_embedding,
    PCA: pca_decomposition,
    ParallelCoordinates: parallel_coordinates,
    RadialVisualizer: radviz,
    CVScores: cv_scores,
    LearningCurve: learning_curve,
    ValidationCurve: validation_curve,
    AlphaSelection: alphas,
    CooksDistance: cooks_distance,
    PredictionError: prediction_error,
    ResidualsPlot: residuals_plot,
    BalancedBinningReference: balanced_binning_reference,
    ClassBalance: class_balance,
    FeatureCorrelation: feature_correlation,
    DispersionPlot: dispersion,
    FreqDistVisualizer: freqdist,
    PosTagVisualizer: postag,
    TSNEVisualizer: tsne,
    UMAPVisualizer: umap,
    GridSearchVisualizer: gridsearch_color_plot,
}

ALIASES = [
    PRCurve,
    KElbow,
    ICDM,
    JointPlotVisualizer,
    PCADecomposition,
    RadViz,
    FrequencyVisualizer,
]

SKIPS = {
    GridSearchColorPlot: "prototype only, no tests",
    ManualAlphaSelection: "prototype only, cannot be instantiated",
    ValidationCurve: "has required positional args: 'param_name', 'param_range'",
    DispersionPlot: "has required positional arg: 'target_words'",
    FrequencyVisualizer: "has required positional arg: 'features'",
}


##########################################################################
## Fixtures
##########################################################################

@pytest.fixture(scope='function')
def figure():
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


def get_model_for_visualizer(Viz):
    """
    Helper function to return the appropriate model for the visualizer class
    """
    if not issubclass(Viz, ModelVisualizer):
        return None

    if issubclass(Viz, ClassificationScoreVisualizer) or Viz is DiscriminationThreshold:
        return GaussianNB

    if issubclass(Viz, RegressionScoreVisualizer):
        if Viz is AlphaSelection:
            return LassoCV
        return Lasso

    if issubclass(Viz, ClusteringScoreVisualizer):
        return MiniBatchKMeans

    if Viz in MODEL_SELECTION_VISUALIZERS:
        return LinearSVC

    if Viz in {RFECV, FeatureImportances}:
        return GaussianNB

    raise TypeError("unknown model for type {}".format(Viz.__name__))


def get_dataset_for_visualizer(Viz):
    """
    Helper function to return the appropriate dataset for the visualizer class
    """
    X, y = None, None

    if issubclass(Viz, RegressionScoreVisualizer):
        X, y = make_regression(random_state=842)

    if issubclass(Viz, ClusteringScoreVisualizer):
        X, y = make_blobs(random_state=112)

    if issubclass(Viz, ClassificationScoreVisualizer):
        X, y = make_classification(random_state=49)

    if Viz in TEXT_VISUALIZERS:
        corpus = load_hobbies()
        X, y = CountVectorizer().fit_transform(corpus.data), corpus.target

    if X is None or y is None:
        # Current default dataset is binary classification
        X, y = make_classification(random_state=23)

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=982)
    return Dataset(Split(X_train, X_test), Split(y_train, y_test))


##########################################################################
## Base API Tests
##########################################################################

@pytest.mark.parametrize("Viz", VISUALIZERS)
def test_instantiation(Viz, figure):
    """
    Ensure all visualizers are instantiated correctly
    """
    if Viz in SKIPS:
        pytest.skip(SKIPS[Viz])

    fig, ax = figure
    kwargs = {
        "ax": ax,
        "fig": fig,
        "size": (9, 6),
        "title": "foo title",
    }

    model = get_model_for_visualizer(Viz)
    if model is not None:
        oz = Viz(model(), **kwargs)
    else:
        oz = Viz(**kwargs)

    assert oz.ax is ax
    assert oz.fig is fig
    assert oz.size == (9, 6)
    assert oz.title == "foo title"


@pytest.mark.skip("too many edge cases, is tested in most visualizer-specific tests")
@pytest.mark.parametrize("Viz", VISUALIZERS)
def test_fit(Viz):
    """
    Ensure that fit returns self and sets up the visualization
    """
    if Viz in SKIPS:
        pytest.skip(SKIPS[Viz])

    kwargs = {
        "ax": mock.MagicMock(),
        "fig": mock.MagicMock(),
    }

    model = get_model_for_visualizer(Viz)
    data = get_dataset_for_visualizer(Viz)
    oz = Viz(model(), **kwargs) if model is not None else Viz(**kwargs)

    assert oz.fit(data.X.train, data.y.train) is oz


@pytest.mark.xfail(reason="quick methods aren't primetime yet")
@pytest.mark.parametrize("Viz, method", list(QUICK_METHODS.items()))
def test_quickmethod(Viz, method):
    """
    Ensures the quick method accepts standard arguments and returns the visualizer
    """
    if Viz in SKIPS:
        pytest.skip(SKIPS[Viz])

    kwargs = {
        "ax": mock.MagicMock(),
        "fig": mock.MagicMock(),
    }

    model = get_model_for_visualizer(Viz)
    pargs = [] if model is None else [model]

    data = get_dataset_for_visualizer(Viz)
    pargs.extend([data.X.train, data.y.train])

    oz = method(*pargs, **kwargs)
    assert isinstance(oz, Viz)
