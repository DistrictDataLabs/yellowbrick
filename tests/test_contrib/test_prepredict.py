# tests.test_contrib.test_prepredict
# Test the prepredict estimator.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Mon Jul 12 07:07:33 2021 -0400
#
# ID: test_prepredict.py [] benjamin@bengfort.com $

"""
Test the prepredict estimator.
"""

##########################################################################
## Imports
##########################################################################

import pytest

from io import BytesIO
from tests.fixtures import Dataset, Split
from tests.base import IS_WINDOWS_OR_CONDA, VisualTestCase

from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.datasets import make_classification, make_regression, make_blobs

from yellowbrick.contrib.prepredict import *
from yellowbrick.regressor import PredictionError
from yellowbrick.classifier import ClassificationReport
import numpy as np

# Set random state
np.random.seed()

##########################################################################
## Fixtures
##########################################################################

@pytest.fixture(scope="class")
def multiclass(request):
    """
    Creates a random multiclass classification dataset fixture
    """
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=8,
        n_redundant=2,
        n_classes=6,
        n_clusters_per_class=3,
        random_state=87,
    )

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=93)

    dataset = Dataset(Split(X_train, X_test), Split(y_train, y_test))
    request.cls.multiclass = dataset


@pytest.fixture(scope="class")
def continuous(request):
    """
    Creates a random continuous regression dataset fixture
    """
    X, y = make_regression(
        n_samples=500,
        n_features=22,
        n_informative=8,
        random_state=42,
        noise=0.2,
        bias=0.2,
    )

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=11)

    # Set a class attribute for regression
    request.cls.continuous = Dataset(Split(X_train, X_test), Split(y_train, y_test))


@pytest.fixture(scope="class")
def blobs(request):
    """
    Create a random blobs clustering dataset fixture
    """
    X, y = make_blobs(
        n_samples=1000, n_features=12, centers=6, shuffle=True, random_state=42
    )

    # Set a class attribute for blobs
    request.cls.blobs = Dataset(X, y)


##########################################################################
## Tests
##########################################################################

@pytest.mark.usefixtures("multiclass")
@pytest.mark.usefixtures("continuous")
@pytest.mark.usefixtures("blobs")
class TestPrePrePredictEstimator(VisualTestCase):
    """
    Pre-predict contrib tests.
    """

    @pytest.mark.xfail(
        IS_WINDOWS_OR_CONDA,
        reason="image comparison failure on Conda 3.8 and 3.9 with RMS 19.307",
    )
    def test_prepredict_classifier(self):
        """
        Test the prepredict estimator with classification report
        """
        # Make prepredictions
        X, y = self.multiclass.X, self.multiclass.y
        y_pred = GaussianNB().fit(X.train, y.train).predict(X.test)

        # Create prepredict estimator with prior predictions
        estimator = PrePredict(y_pred, CLASSIFIER)
        assert estimator.fit(X.train, y.train) is estimator
        assert estimator.predict(X.train) is y_pred
        assert estimator.score(X.test, y.test) == pytest.approx(0.41, rel=1e-3)

        # Test that a visualizer works with the pre-predictions.
        viz = ClassificationReport(estimator)
        viz.fit(None, y.train)
        viz.score(None, y.test)
        viz.finalize()

        self.assert_images_similar(viz)

    def test_prepredict_regressor(self):
        """
        Test the prepredict estimator with a prediction error plot
        """
        # Make prepredictions
        X, y = self.continuous.X, self.continuous.y
        y_pred = LinearRegression().fit(X.train, y.train).predict(X.test)

        # Create prepredict estimator with prior predictions
        estimator = PrePredict(y_pred, REGRESSOR)
        assert estimator.fit(X.train, y.train) is estimator
        assert estimator.predict(X.train) is y_pred
        assert estimator.score(X.test, y.test) == pytest.approx(0.9999983124154966, rel=1e-2)

        # Test that a visualizer works with the pre-predictions.
        viz = PredictionError(estimator)
        viz.fit(X.train, y.train)
        viz.score(X.test, y.test)
        viz.finalize()

        self.assert_images_similar(viz, tol=10.0)

    def test_prepredict_clusterer(self):
        """
        Test the prepredict estimator with a silhouette visualizer
        """
        X = self.blobs.X
        y_pred = MiniBatchKMeans(random_state=831).fit(X).predict(X)

         # Create prepredict estimator with prior predictions
        estimator = PrePredict(y_pred, CLUSTERER)
        assert estimator.fit(X) is estimator
        assert estimator.predict(X) is y_pred
        assert estimator.score(X) == pytest.approx(0.5477478541994333, rel=1e-2)

        # NOTE: there is currently no cluster visualizer that can take advantage of
        # the prepredict utility since they all require learned attributes.

    def test_load(self):
        """
        Test the various ways that prepredict loads data
        """
        # Test callable
        ppe = PrePredict(lambda: self.multiclass.y.test)
        assert ppe._load() is self.multiclass.y.test

        # Test file-like object, assume that str and pathlib.Path work similarly
        f = BytesIO()
        np.save(f, self.continuous.y.test)
        f.seek(0)
        ppe = PrePredict(f)
        assert np.array_equal(ppe._load(), self.continuous.y.test)

        # Test direct array-like completed in other tests.
