# tests.test_features.test_importances
# Test the feature importance visualizers
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Fri Mar 02 15:23:22 2018 -0500
#
# Copyright (C) 2018 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_importances.py [] benjamin@bengfort.com $

"""
Test the feature importance visualizers
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from yellowbrick.exceptions import NotFitted
from yellowbrick.features.importances import *

from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin

try:
    from unittest import mock
except ImportError:
    import mock

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Feature Importances Tests
##########################################################################

class TestFeatureImportancesVisualizer(VisualTestCase, DatasetMixin):
    """
    FeatureImportances visualizer
    """

    def test_integration_feature_importances(self):
        """
        Integration test of visualizer with feature importances param
        """

        occupancy = self.load_data('occupancy')
        features = [
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]

        # Extract X and y as numpy arrays
        X = occupancy[features].copy()
        X = X.view((float, len(X.dtype.names)))
        y = occupancy['occupancy'].astype(int)

        fig = plt.figure()
        ax = fig.add_subplot()

        clf = GradientBoostingClassifier(random_state=42)
        viz = FeatureImportances(clf, ax=ax)
        viz.fit(X, y)
        viz.poof()

        self.assert_images_similar(viz)

    def test_integration_coef(self):
        """
        Integration test of visualizer with coef param
        """

        concrete = self.load_data('concrete')
        feats = ['cement','slag','ash','water','splast','coarse','fine','age']

        # Create X and y datasets as numpy arrays
        X = concrete[feats].copy()
        X = X.view((float, len(X.dtype.names)))
        y = concrete['strength']

        fig = plt.figure()
        ax = fig.add_subplot()

        reg = Lasso(random_state=42)
        feats = list(map(lambda s: s.title(), feats))
        viz = FeatureImportances(reg, ax=ax, labels=feats, relative=False)
        viz.fit(X, y)
        viz.poof()

        self.assert_images_similar(viz)

    def test_integration_quick_method(self):
        """
        Integration test of quick method
        """

        occupancy = self.load_data('occupancy')
        features = [
            "temperature", "relative_humidity", "light", "C02", "humidity"
        ]

        # Create X and y datasets as numpy arrays
        X = occupancy[features].copy()
        X = X.view((float, len(X.dtype.names)))
        y = occupancy['occupancy'].astype(int)

        fig = plt.figure()
        ax = fig.add_subplot()

        clf = RandomForestClassifier(random_state=42)
        g = feature_importances(clf, X, y, ax)

        self.assert_images_similar(ax=g)

    def test_fit_no_importances_model(self):
        """
        Fitting a model without feature importances raises an exception
        """
        X = np.random.rand(100, 42)
        y = np.random.rand(100)

        visualizer = FeatureImportances(MockEstimator())
        expected_error = "could not find feature importances param on MockEstimator"

        with pytest.raises(YellowbrickTypeError, match=expected_error):
            visualizer.fit(X, y)

    def test_fit_sorted_params(self):
        """
        On fit, sorted features_ and feature_importances_ params are created
        """
        coefs = np.array([0.4, 0.2, 0.08, 0.07, 0.16, .23, 0.38, 0.1, 0.05])
        names = np.array(['a', 'b',  'c',  'd',  'e', 'f',  'g', 'h',  'i'])

        model = MockEstimator()
        model.make_importance_param(value=coefs)

        visualizer = FeatureImportances(model, labels=names)
        visualizer.fit(np.random.rand(100, len(names)), np.random.rand(100))

        assert hasattr(visualizer, 'features_')
        assert hasattr(visualizer, 'feature_importances_')

        # get the expected sort index
        sort_idx = np.argsort(coefs)

        # assert sorted
        npt.assert_array_equal(names[sort_idx], visualizer.features_)
        npt.assert_array_equal(coefs[sort_idx], visualizer.feature_importances_)

    def test_fit_relative(self):
        """
        Test fit computes relative importances
        """
        coefs = np.array([0.4, 0.2, 0.08, 0.07, 0.16, .23, 0.38, 0.1, 0.05])

        model = MockEstimator()
        model.make_importance_param(value=coefs)

        visualizer = FeatureImportances(model, relative=True)
        visualizer.fit(np.random.rand(100, len(coefs)), np.random.rand(100))

        expected = 100.0 * coefs / coefs.max()
        expected.sort()
        npt.assert_array_equal(visualizer.feature_importances_, expected)

    def test_fit_not_relative(self):
        """
        Test fit stores unmodified importances
        """
        coefs = np.array([0.4, 0.2, 0.08, 0.07, 0.16, .23, 0.38, 0.1, 0.05])

        model = MockEstimator()
        model.make_importance_param(value=coefs)

        visualizer = FeatureImportances(model, relative=False)
        visualizer.fit(np.random.rand(100, len(coefs)), np.random.rand(100))

        coefs.sort()
        npt.assert_array_equal(visualizer.feature_importances_, coefs)

    def test_fit_absolute(self):
        """
        Test fit with absolute values
        """
        coefs = np.array([0.4, 0.2, -0.08, 0.07, 0.16, .23, -0.38, 0.1, -0.05])

        model = MockEstimator()
        model.make_importance_param(value=coefs)

        # Test absolute value
        visualizer = FeatureImportances(model, absolute=True, relative=False)
        visualizer.fit(np.random.rand(100, len(coefs)), np.random.rand(100))

        expected = np.array([0.05, 0.07, 0.08, 0.1, 0.16, 0.2, .23, 0.38, 0.4])
        npt.assert_array_equal(visualizer.feature_importances_, expected)

        # Test no absolute value
        visualizer = FeatureImportances(model, absolute=False, relative=False)
        visualizer.fit(np.random.rand(100, len(coefs)), np.random.rand(100))

        expected = np.array([-0.38, -0.08, -0.05, 0.07, 0.1, 0.16, 0.2, .23, 0.4])
        npt.assert_array_equal(visualizer.feature_importances_, expected)


    @pytest.mark.skipif(pd is None, reason="pandas is required for this test")
    def test_fit_dataframe(self):
        """
        Ensure feature names are extracted from DataFrame columns
        """
        labels = ['a', 'b', 'c', 'd', 'e', 'f']
        df = pd.DataFrame(np.random.rand(100, 6), columns=labels)
        s = pd.Series(np.random.rand(100), name='target')

        assert df.shape == (100, 6)

        model = MockEstimator()
        model.make_importance_param(value=np.linspace(0, 1, 6))

        visualizer = FeatureImportances(model)
        visualizer.fit(df, s)

        assert hasattr(visualizer, 'features_')
        npt.assert_array_equal(visualizer.features_, np.array(df.columns))

    def test_fit_makes_labels(self):
        """
        Assert that the fit process makes label indices
        """
        model = MockEstimator()
        model.make_importance_param(value=np.linspace(0, 1, 10))

        visualizer = FeatureImportances(model)
        visualizer.fit(np.random.rand(100, 10), np.random.rand(100))

        # Don't have to worry about label space since importances are linspace
        assert hasattr(visualizer, 'features_')
        npt.assert_array_equal(np.arange(10), visualizer.features_)

    def test_fit_calls_draw(self):
        """
        Assert that fit calls draw
        """
        model = MockEstimator()
        model.make_importance_param('coef_')

        visualizer = FeatureImportances(model)

        with mock.patch.object(visualizer, 'draw') as mdraw:
            visualizer.fit(np.random.rand(100,42), np.random.rand(100))
            mdraw.assert_called_once()

    def test_draw_raises_unfitted(self):
        """
        Assert draw raises exception when not fitted
        """
        visualizer = FeatureImportances(Lasso())
        with pytest.raises(NotFitted):
            visualizer.draw()

    def test_find_importances_param(self):
        """
        Test the expected parameters can be found
        """
        params = ('feature_importances_', 'coef_')

        for param in params:
            model = MockEstimator()
            model.make_importance_param(param, 'foo')
            visualizer = FeatureImportances(model)

            assert hasattr(model, param), "expected '{}' missing".format(param)
            for oparam in params:
                if oparam == param: continue
                assert not hasattr(model, oparam), "unexpected '{}'".format(oparam)

            importances = visualizer._find_importances_param()
            assert importances == 'foo'

    def test_find_importances_param_priority(self):
        """
        With both feature_importances_ and coef_, one has priority
        """
        model = MockEstimator()
        model.make_importance_param('feature_importances_', 'foo')
        model.make_importance_param('coef_', 'bar')
        visualizer = FeatureImportances(model)

        assert hasattr(model, 'feature_importances_')
        assert hasattr(model, 'coef_')

        importances = visualizer._find_importances_param()
        assert importances == 'foo'

    def test_find_importances_param_not_found(self):
        """
        Raises an exception when importances param not found
        """
        model = MockEstimator()
        visualizer = FeatureImportances(model)

        assert not hasattr(model, 'feature_importances_')
        assert not hasattr(model, 'coef_')

        with pytest.raises(YellowbrickTypeError):
            visualizer._find_importances_param()

    def test_xlabel(self):
        """
        Check the various xlabels are sensical
        """
        model = MockEstimator()
        model.make_importance_param('feature_importances_')
        visualizer = FeatureImportances(model, xlabel="foo", relative=True)

        # Assert the visualizer uses the user supplied xlabel
        assert visualizer._get_xlabel() == "foo", "could not set user xlabel"

        # Check the visualizer default relative xlabel
        visualizer.set_params(xlabel=None)
        assert "relative" in visualizer._get_xlabel()

        # Check value xlabel with default
        visualizer.set_params(relative=False)
        assert "relative" not in visualizer._get_xlabel()

        # Check coeficients
        model = MockEstimator()
        model.make_importance_param('coef_')
        visualizer = FeatureImportances(model, xlabel="baz", relative=True)

        # Assert the visualizer uses the user supplied xlabel
        assert visualizer._get_xlabel() == "baz", "could not set user xlabel"

        # Check the visualizer default relative xlabel
        visualizer.set_params(xlabel=None)
        assert "coefficient" in visualizer._get_xlabel()
        assert "relative" in visualizer._get_xlabel()

        # Check value xlabel with default
        visualizer.set_params(relative=False)
        assert "coefficient" in visualizer._get_xlabel()
        assert "relative" not in visualizer._get_xlabel()


    def test_is_fitted(self):
        """
        Test identification if is fitted
        """
        visualizer = FeatureImportances(Lasso())
        assert not visualizer._is_fitted()

        visualizer.features_ = "foo"
        assert not visualizer._is_fitted()

        visualizer.feature_importances_ = "bar"
        assert visualizer._is_fitted()

        del visualizer.features_
        assert not visualizer._is_fitted()


##########################################################################
## Mock Estimator
##########################################################################

class MockEstimator(BaseEstimator):
    """
    Creates params when fit is called on demand.
    """

    def make_importance_param(self, name='feature_importances_', value=None):
        if value is None:
            value = np.random.rand(42)
        setattr(self, name, value)

    def fit(self, X, y=None, **kwargs):
        return self
