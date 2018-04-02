# tests.test_classifier.test_classification_report
# Tests for the classification report visualizer
#
# Author:  Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Sun Mar 18 16:57:27 2018 -0400
#
# ID: test_classification_report.py [] benjamin@bengfort.com $

"""
Tests for the classification report visualizer
"""

##########################################################################
## Imports
##########################################################################

import pytest
import yellowbrick as yb
import matplotlib.pyplot as plt

from collections import namedtuple

from yellowbrick.classifier.classification_report import *

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LassoCV, LogisticRegression

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Fixtures
##########################################################################

# Helpers for fixtures
Dataset = namedtuple('Dataset', 'X,y')
Split = namedtuple('Split', 'train,test')


@pytest.fixture(scope='class')
def binary(request):
    """
    Creates a random binary classification dataset fixture
    """
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=8, n_redundant=2,
        n_classes=2, n_clusters_per_class=3, random_state=87
    )

    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.2, random_state=93
    )

    dataset = Dataset(Split(X_train, X_test), Split(y_train, y_test))
    request.cls.binary = dataset


@pytest.fixture(scope='class')
def multiclass(request):
    """
    Creates a random multiclass classification dataset fixture
    """
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=8, n_redundant=2,
        n_classes=6, n_clusters_per_class=3, random_state=87
    )

    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.2, random_state=93
    )

    dataset = Dataset(Split(X_train, X_test), Split(y_train, y_test))
    request.cls.multiclass = dataset



##########################################################################
##  Test for Classification Report
##########################################################################

@pytest.mark.usefixtures("binary", "multiclass")
class ClassificationReportTests(VisualTestCase, DatasetMixin):
    """
    ClassificationReport visualizer tests
    """

    def test_binary_class_report(self):
        """
        Correctly generates a report for binary classification with LinearSVC
        """
        _, ax = plt.subplots()

        viz = ClassificationReport(LinearSVC(), ax=ax)
        viz.fit(self.binary.X.train, self.binary.y.train)
        viz.score(self.binary.X.test, self.binary.y.test)

        self.assert_images_similar(viz)

        assert viz.scores_ == {
            'precision': {0: 0.7446808510638298, 1: 0.8490566037735849},
            'recall': {0: 0.813953488372093, 1: 0.7894736842105263},
            'f1': {0: 0.7777777777777778, 1: 0.8181818181818182}
        }

    def test_multiclass_class_report(self):
        """
        Correctly generates report for multi-class with LogisticRegression
        """
        _, ax = plt.subplots()

        viz = ClassificationReport(LogisticRegression(random_state=12), ax=ax)
        viz.fit(self.multiclass.X.train, self.multiclass.y.train)
        viz.score(self.multiclass.X.test, self.multiclass.y.test)

        self.assert_images_similar(viz)

        assert viz.scores_ == {
            'precision': {
                0: 0.5333333333333333, 1: 0.5, 2: 0.45,
                3: 0.4, 4: 0.4, 5: 0.5882352941176471
            }, 'recall': {
                0: 0.42105263157894735, 1: 0.5625, 2: 0.6428571428571429,
                3: 0.3157894736842105, 4: 0.375, 5: 0.625
            }, 'f1': {
                0: 0.47058823529411764, 1: 0.5294117647058824,
                2: 0.5294117647058824, 3: 0.35294117647058826,
                4: 0.38709677419354843, 5: 0.6060606060606061
            }}

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_integration(self):
        """
        Test with Pandas DataFrame and Series input
        """
        _, ax = plt.subplots()

        # Load the occupancy dataset from fixtures
        data = self.load_data('occupancy')
        target = 'occupancy'
        features = [
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]

        # Create instances and target
        X = pd.DataFrame(data[features])
        y = pd.Series(data[target].astype(int))

        # Create train/test splits
        splits = tts(X, y, test_size=0.2, random_state=4512)
        X_train, X_test, y_train, y_test = splits

        classes = ['unoccupied', 'occupied']

        # Create classification report
        model = GaussianNB()
        viz = ClassificationReport(model, ax=ax, classes=classes)
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)

        self.assert_images_similar(viz, tol=0.1)

        # Ensure correct classification scores under the hood
        assert viz.scores_ == {
            'precision': {
                'unoccupied': 0.999347471451876,
                'occupied': 0.8825214899713467
            }, 'recall': {
                'unoccupied': 0.9613935969868174,
                'occupied': 0.9978401727861771
            }, 'f1': {
                'unoccupied': 0.9800031994880819,
                'occupied': 0.9366447034972124
            }}

    @pytest.mark.skip(reason="requires random state in quick method")
    def test_quick_method(self):
        """
        Test the quick method with a random dataset
        """
        X, y = make_classification(
            n_samples=400, n_features=20, n_informative=8, n_redundant=8,
            n_classes=2, n_clusters_per_class=4, random_state=27
        )

        _, ax = plt.subplots()
        classification_report(DecisionTreeClassifier(), X, y, ax=ax)

        self.assert_images_similar(ax=ax)

    def test_isclassifier(self):
        """
        Assert that only classifiers can be used with the visualizer.
        """

        message = (
            'This estimator is not a classifier; '
            'try a regression or clustering score visualizer instead!'
        )

        with pytest.raises(yb.exceptions.YellowbrickError, match=message):
            ClassificationReport(LassoCV())
