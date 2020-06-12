# tests.test_classifier.test_threshold
# Ensure that the discrimination threshold visualizations work.
#
# Author:  Nathan Danielsen
# Author:  Benjamin Bengfort
# Created: Wed April 26 20:17:29 2017 -0700
#
# Copyright (C) 2017 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_threshold.py [] nathan.danielsen@gmail.com $

"""
Ensure that the DiscriminationThreshold visualizations work.
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest
import yellowbrick as yb
import matplotlib.pyplot as plt

from yellowbrick.classifier.threshold import *
from yellowbrick.datasets import load_occupancy
from yellowbrick.utils import is_probabilistic, is_classifier

from unittest.mock import patch
from tests.base import VisualTestCase
from numpy.testing import assert_array_equal

from sklearn.base import ClassifierMixin
from sklearn.svm import LinearSVC, NuSVC
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import pandas as pd
except ImportError:
    pd = None

##########################################################################
## DiscriminationThreshold Test Cases
##########################################################################


class TestDiscriminationThreshold(VisualTestCase):
    """
    DiscriminationThreshold visualizer tests
    """

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_binary_discrimination_threshold(self):
        """
        Correctly generates viz for binary classification with BernoulliNB
        """
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=8,
            n_redundant=8,
            n_classes=2,
            n_clusters_per_class=4,
            random_state=854,
        )

        _, ax = plt.subplots()

        model = BernoulliNB(3)
        visualizer = DiscriminationThreshold(model, ax=ax, random_state=23)

        visualizer.fit(X, y)
        visualizer.finalize()

        self.assert_images_similar(visualizer)

    def test_multiclass_discrimination_threshold(self):
        """
        Assert exception is raised in multiclass case.
        """
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=8,
            n_redundant=8,
            n_classes=3,
            n_clusters_per_class=4,
            random_state=854,
        )

        visualizer = DiscriminationThreshold(GaussianNB(), random_state=23)
        msg = "multiclass format is not supported"

        with pytest.raises(ValueError, match=msg):
            visualizer.fit(X, y)

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_integration(self):
        """
        Test with Pandas DataFrame and Series input
        """
        _, ax = plt.subplots()

        # Load the occupancy dataset from fixtures
        data = load_occupancy(return_dataset=True)
        X, y = data.to_pandas()

        classes = ["unoccupied", "occupied"]

        # Create the visualizer
        viz = DiscriminationThreshold(
            LogisticRegression(), ax=ax, classes=classes, random_state=193
        )
        viz.fit(X, y)
        viz.finalize()

        self.assert_images_similar(viz, tol=0.1)

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_numpy_integration(self):
        """
        Test with NumPy arrays
        """
        _, ax = plt.subplots()

        # Load the occupancy dataset from fixtures
        data = load_occupancy(return_dataset=True)
        X, y = data.to_numpy()

        classes = ["unoccupied", "occupied"]

        # Create the visualizer
        viz = DiscriminationThreshold(
            LogisticRegression(), ax=ax, classes=classes, random_state=193
        )
        viz.fit(X, y)
        viz.finalize()

        self.assert_images_similar(viz, tol=0.1)

    def test_quick_method(self):
        """
        Test for thresholdviz quick method with random dataset
        """

        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=8,
            n_redundant=8,
            n_classes=2,
            n_clusters_per_class=4,
            random_state=2721,
        )

        _, ax = plt.subplots()

        discrimination_threshold(BernoulliNB(3), X, y, ax=ax, random_state=5, show=False)
        self.assert_images_similar(ax=ax, tol=10)

    @patch.object(DiscriminationThreshold, "draw", autospec=True)
    def test_fit(self, mock_draw):
        """
        Test the fit method generates scores, calls draw, and returns self
        """
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=8,
            n_redundant=8,
            n_classes=2,
            n_clusters_per_class=4,
            random_state=1221,
        )

        visualizer = DiscriminationThreshold(BernoulliNB())
        assert not hasattr(visualizer, "thresholds_")
        assert not hasattr(visualizer, "cv_scores_")

        out = visualizer.fit(X, y)

        assert out is visualizer
        mock_draw.assert_called_once()
        assert hasattr(visualizer, "thresholds_")
        assert hasattr(visualizer, "cv_scores_")

        for metric in METRICS:
            assert metric in visualizer.cv_scores_
            assert "{}_lower".format(metric) in visualizer.cv_scores_
            assert "{}_upper".format(metric) in visualizer.cv_scores_

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_binary_discrimination_threshold_alt_args(self):
        """
        Correctly generates visualization with alternate arguments
        """
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=10,
            n_redundant=3,
            n_classes=2,
            n_clusters_per_class=4,
            random_state=1231,
            flip_y=0.1,
            weights=[0.35, 0.65],
        )

        exclude = ["queue_rate", "fscore"]
        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        visualizer = DiscriminationThreshold(
            NuSVC(), exclude=exclude, cv=cv, random_state=98239
        )

        visualizer.fit(X, y)
        visualizer.finalize()

        for metric in exclude:
            assert metric not in visualizer.cv_scores_
            assert "{}_lower".format(metric) not in visualizer.cv_scores_
            assert "{}_upper".format(metric) not in visualizer.cv_scores_

        self.assert_images_similar(visualizer)

    def test_threshold_default_initialization(self):
        """
        Test initialization default parameters
        """
        model = BernoulliNB(3)
        viz = DiscriminationThreshold(model)
        assert viz.estimator is model
        assert viz.color is None
        assert viz.title is None
        assert viz.n_trials == 50
        assert viz.cv == 0.1
        assert_array_equal(viz.quantiles, np.array((0.1, 0.5, 0.9)))

    def test_requires_classifier(self):
        """
        Assert requires a classifier
        """
        message = "requires a probabilistic binary classifier"
        assert not is_classifier(Ridge)

        with pytest.raises(yb.exceptions.YellowbrickError, match=message):
            DiscriminationThreshold(Ridge())

    def test_requires_probabilistic_classifier(self):
        """
        Assert requires probabilistic classifier
        """
        message = "requires a probabilistic binary classifier"

        class RoboClassifier(ClassifierMixin):
            """
            Dummy Non-Probabilistic Classifier
            """

            def fit(self, X, y):
                self.classes_ = [0, 1]
                return self

        assert is_classifier(RoboClassifier)
        assert not is_probabilistic(RoboClassifier)

        with pytest.raises(yb.exceptions.YellowbrickError, match=message):
            DiscriminationThreshold(RoboClassifier())

    def test_accepts_predict_proba(self):
        """
        Will accept classifiers with predict proba function
        """
        model = RandomForestClassifier
        assert is_classifier(model)
        assert is_probabilistic(model)
        assert not hasattr(model, "decision_function")
        assert hasattr(model, "predict_proba")

        try:
            DiscriminationThreshold(model())
        except YellowbrickTypeError:
            pytest.fail("did not accept decision function model")

    def test_accepts_decision_function(self):
        """
        Will accept classifiers with decision function
        """
        model = LinearSVC
        assert is_classifier(model)
        assert is_probabilistic(model)
        assert hasattr(model, "decision_function")
        assert not hasattr(model, "predict_proba")

        try:
            DiscriminationThreshold(model())
        except YellowbrickTypeError:
            pytest.fail("did not accept decision function model")

    def test_bad_quantiles(self):
        """
        Assert exception is raised when bad quantiles are passed in.
        """
        msg = (
            "quantiles must be a sequence of three "
            "monotonically increasing values less than 1"
        )

        with pytest.raises(YellowbrickValueError, match=msg):
            DiscriminationThreshold(NuSVC(), quantiles=[0.25, 0.1, 0.75])

    def test_bad_cv(self):
        """
        Assert an exception is raised when a bad cv value is passed in
        """
        with pytest.raises(YellowbrickValueError, match="not a valid cv splitter"):
            DiscriminationThreshold(NuSVC(), cv="foo")

    def test_splitter_random_state(self):
        """
        Test splitter random state is modified
        """
        viz = DiscriminationThreshold(NuSVC(), random_state=None)
        assert viz._check_cv(None, random_state=None).random_state is None
        assert viz._check_cv(None, random_state=42).random_state == 42

        splits = StratifiedShuffleSplit(n_splits=1, random_state=None)
        assert viz._check_cv(splits, random_state=None).random_state is None
        assert viz._check_cv(splits, random_state=23).random_state == 23

        splits = StratifiedShuffleSplit(n_splits=1, random_state=181)
        assert viz._check_cv(splits, random_state=None).random_state == 181
        assert viz._check_cv(splits, random_state=72).random_state == 72

    def test_bad_exclude(self):
        """
        Assert an exception is raised on bad exclude param
        """
        with pytest.raises(YellowbrickValueError, match="not a valid metric"):
            DiscriminationThreshold(NuSVC(), exclude="foo")

        with pytest.raises(YellowbrickValueError, match="not a valid metric"):
            DiscriminationThreshold(NuSVC(), exclude=["queue_rate", "foo"])
