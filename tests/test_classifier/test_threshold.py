# tests.test_classifier.test_threshold
# Ensure that the discrimination threshold visualizations work.
#
# Author:  Nathan Danielsen <ndanielsen@gmail.com>
# Author:  Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created: Wed April 26 20:17:29 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_threshold.py [] nathan.danielsen@gmail.com $

"""
Ensure that the DiscriminationThreshold visualizations work.
"""

##########################################################################
## Imports
##########################################################################

import pytest
import yellowbrick as yb
import matplotlib.pyplot as plt

from yellowbrick.classifier.threshold import *

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin

from sklearn.linear_model import Ridge
from sklearn.datasets import make_classification
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.model_selection import train_test_split as tts

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## DiscriminationThreshold Test Cases
##########################################################################

@pytest.mark.usefixtures("binary", "multiclass")
class TestDiscriminationThreshold(VisualTestCase, DatasetMixin):
    """
    DiscriminationThreshold visualizer tests
    """

    def test_threshold_default_initialization(self):
        """
        Test initialization default parameters
        """
        model = BernoulliNB(3)
        viz = DiscriminationThreshold(model)
        assert viz.estimator is model
        assert viz.color is None
        assert viz.title is None
        assert viz.plot_data is None
        assert viz.n_trials == 50
        assert viz.test_size_percent == 0.1
        assert viz.quantiles == (0.1, 0.5, 0.9)

    def test_binary_discrimination_threshold(self):
        """
        Correctly generates viz for binary classification with BernoulliNB
        """
        _, ax = plt.subplots()

        model = BernoulliNB(3)
        visualizer = DiscriminationThreshold(model, ax=ax, random_state=23)

        visualizer.fit(self.binary.X.train, self.binary.y.train)
        visualizer.poof()

        self.assert_images_similar(visualizer)

    def test_multiclass_discrimination_threshold(self):
        """
        Assert exception is raised in multiclass case.
        """
        visualizer = DiscriminationThreshold(GaussianNB(), random_state=23)

        with pytest.raises(ValueError, match="multiclass format is not supported"):
            visualizer.fit(self.multiclass.X.train, self.multiclass.y.train)

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
        viz = DiscriminationThreshold(
            model, ax=ax, classes=classes, random_state=193
        )
        viz.fit(X_train, y_train)
        viz.poof()

        self.assert_images_similar(viz, tol=0.1)

    def test_quick_method(self):
        """
        Test for thresholdviz quick method with random dataset
        """

        X, y = make_classification(
            n_samples=400, n_features=20, n_informative=8, n_redundant=8,
            n_classes=2, n_clusters_per_class=4, random_state=2721
        )

        _, ax = plt.subplots()

        thresholdviz(BernoulliNB(3), X, y, ax=ax)
        self.assert_images_similar(ax=ax)

    def test_isclassifier(self):
        """
        Assert only classifiers can be used with the visualizer
        """
        message = (
            'This estimator is not a classifier; '
            'try a regression or clustering score visualizer instead!'
        )

        with pytest.raises(yb.exceptions.YellowbrickError, match=message):
            DiscriminationThreshold(Ridge())
