# tests.test_model_selection.test_importances
# Test the feature importance visualizers
#
# Author:  Benjamin Bengfort
# Author:  Rebecca Bilbro
# Created: Fri Mar 02 15:23:22 2018 -0500
#
# Copyright (C) 2018 The scikit-yb developers
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
from yellowbrick.model_selection.importances import *
from yellowbrick.datasets import load_occupancy, load_concrete

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, Lasso

from unittest import mock
from tests.base import VisualTestCase

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Feature Importances Tests
##########################################################################


class TestFeatureImportancesVisualizer(VisualTestCase):
    """
    Test FeatureImportances visualizer
    """

    def test_integration_feature_importances(self):
        """
        Integration test of visualizer with feature importances param
        """

        # Load the test dataset
        X, y = load_occupancy(return_dataset=True).to_numpy()

        fig = plt.figure()
        ax = fig.add_subplot()

        clf = GradientBoostingClassifier(random_state=42)
        viz = FeatureImportances(clf, ax=ax)
        viz.fit(X, y)
        viz.finalize()

        # Appveyor and Linux conda non-text-based differences
        self.assert_images_similar(viz, tol=13.0)

    def test_integration_coef(self):
        """
        Integration test of visualizer with coef param
        """

        # Load the test dataset
        dataset = load_concrete(return_dataset=True)
        X, y = dataset.to_numpy()
        features = dataset.meta["features"]

        fig = plt.figure()
        ax = fig.add_subplot()

        reg = Lasso(random_state=42)
        features = list(map(lambda s: s.title(), features))
        viz = FeatureImportances(reg, ax=ax, labels=features, relative=False)
        viz.fit(X, y)
        viz.finalize()

        # Appveyor and Linux conda non-text-based differences
        self.assert_images_similar(viz, tol=16.2)

    def test_integration_quick_method(self):
        """
        Integration test of quick method
        """

        # Load the test dataset
        X, y = load_occupancy(return_dataset=True).to_numpy()

        fig = plt.figure()
        ax = fig.add_subplot()

        clf = RandomForestClassifier(random_state=42)
        g = feature_importances(clf, X, y, ax=ax, show=False)

        # Appveyor and Linux conda non-text-based differences
        self.assert_images_similar(g, tol=15.0)

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
        coefs = np.array([0.4, 0.2, 0.08, 0.07, 0.16, 0.23, 0.38, 0.1, 0.05])
        names = np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"])

        model = MockEstimator()
        model.make_importance_param(value=coefs)

        visualizer = FeatureImportances(model, labels=names)
        visualizer.fit(np.random.rand(100, len(names)), np.random.rand(100))

        assert hasattr(visualizer, "features_")
        assert hasattr(visualizer, "feature_importances_")

        # get the expected sort index
        sort_idx = np.argsort(coefs)

        # assert sorted
        npt.assert_array_equal(names[sort_idx], visualizer.features_)
        npt.assert_array_equal(coefs[sort_idx], visualizer.feature_importances_)

    def test_fit_relative(self):
        """
        Test fit computes relative importances
        """
        coefs = np.array([0.4, 0.2, 0.08, 0.07, 0.16, 0.23, 0.38, 0.1, 0.05])

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
        coefs = np.array([0.4, 0.2, 0.08, 0.07, 0.16, 0.23, 0.38, 0.1, 0.05])

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
        coefs = np.array([0.4, 0.2, -0.08, 0.07, 0.16, 0.23, -0.38, 0.1, -0.05])

        model = MockEstimator()
        model.make_importance_param(value=coefs)

        # Test absolute value
        visualizer = FeatureImportances(model, absolute=True, relative=False)
        visualizer.fit(np.random.rand(100, len(coefs)), np.random.rand(100))

        expected = np.array([0.05, 0.07, 0.08, 0.1, 0.16, 0.2, 0.23, 0.38, 0.4])
        npt.assert_array_equal(visualizer.feature_importances_, expected)

        # Test no absolute value
        visualizer = FeatureImportances(model, absolute=False, relative=False)
        visualizer.fit(np.random.rand(100, len(coefs)), np.random.rand(100))

        expected = np.array([-0.38, -0.08, -0.05, 0.07, 0.1, 0.16, 0.2, 0.23, 0.4])
        npt.assert_array_equal(visualizer.feature_importances_, expected)

    def test_multi_coefs(self):
        """
        Test fit with multidimensional coefficients and stack warning
        """
        coefs = np.array(
            [
                [0.4, 0.2, -0.08, 0.07, 0.16, 0.23, -0.38, 0.1, -0.05],
                [0.41, 0.12, -0.1, 0.1, 0.14, 0.21, 0.01, 0.31, -0.15],
                [0.31, 0.2, -0.01, 0.1, 0.22, 0.23, 0.01, 0.12, -0.15],
            ]
        )

        model = MockEstimator()
        model.make_importance_param(value=coefs)

        visualizer = FeatureImportances(model, stack=False)

        with pytest.warns(YellowbrickWarning):
            visualizer.fit(
                np.random.rand(100, len(np.mean(coefs, axis=0))), np.random.rand(100)
            )

        npt.assert_equal(visualizer.feature_importances_.ndim, 1)

    def test_multi_coefs_stacked(self):
        """
        Test stack plot with multidimensional coefficients
        """
        X, y = load_iris(True)

        viz = FeatureImportances(
            LogisticRegression(solver="liblinear", random_state=222), stack=True
        )
        viz.fit(X, y)
        viz.finalize()

        npt.assert_equal(viz.feature_importances_.shape, (3, 4))
        # Appveyor and Linux conda non-text-based differences
        self.assert_images_similar(viz, tol=17.5)

    @pytest.mark.skipif(pd is None, reason="pandas is required for this test")
    def test_fit_dataframe(self):
        """
        Ensure feature names are extracted from DataFrame columns
        """
        labels = ["a", "b", "c", "d", "e", "f"]
        df = pd.DataFrame(np.random.rand(100, 6), columns=labels)
        s = pd.Series(np.random.rand(100), name="target")

        assert df.shape == (100, 6)

        model = MockEstimator()
        model.make_importance_param(value=np.linspace(0, 1, 6))

        visualizer = FeatureImportances(model)
        visualizer.fit(df, s)

        assert hasattr(visualizer, "features_")
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
        assert hasattr(visualizer, "features_")
        npt.assert_array_equal(np.arange(10), visualizer.features_)

    def test_fit_calls_draw(self):
        """
        Assert that fit calls draw
        """
        model = MockEstimator()
        model.make_importance_param("coef_")

        visualizer = FeatureImportances(model)

        with mock.patch.object(visualizer, "draw") as mdraw:
            visualizer.fit(np.random.rand(100, 42), np.random.rand(100))
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
        params = ("feature_importances_", "coef_")

        for param in params:
            model = MockEstimator()
            model.make_importance_param(param, "foo")
            visualizer = FeatureImportances(model)

            assert hasattr(model, param), "expected '{}' missing".format(param)
            for oparam in params:
                if oparam == param:
                    continue
                assert not hasattr(model, oparam), "unexpected '{}'".format(oparam)

            importances = visualizer._find_importances_param()
            assert importances == "foo"

    def test_find_importances_param_priority(self):
        """
        With both feature_importances_ and coef_, one has priority
        """
        model = MockEstimator()
        model.make_importance_param("feature_importances_", "foo")
        model.make_importance_param("coef_", "bar")
        visualizer = FeatureImportances(model)

        assert hasattr(model, "feature_importances_")
        assert hasattr(model, "coef_")

        importances = visualizer._find_importances_param()
        assert importances == "foo"

    def test_find_importances_param_not_found(self):
        """
        Raises an exception when importances param not found
        """
        model = MockEstimator()
        visualizer = FeatureImportances(model)

        assert not hasattr(model, "feature_importances_")
        assert not hasattr(model, "coef_")

        with pytest.raises(YellowbrickTypeError):
            visualizer._find_importances_param()

    def test_find_classes_param_not_found(self):
        """
        Raises an exception when classes param not found
        """
        model = MockClassifier()
        visualizer = FeatureImportances(model)

        assert not hasattr(model, "classes_")

        e = "could not find classes_ param on {}".format(
            visualizer.estimator.__class__.__name__
        )
        with pytest.raises(YellowbrickTypeError, match=e):
            visualizer._find_classes_param()

    def test_xlabel(self):
        """
        Check the various xlabels are sensical
        """
        model = MockEstimator()
        model.make_importance_param("feature_importances_")
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
        model.make_importance_param("coef_")
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

    def test_with_fitted(self):
        """
        Test that visualizer properly handles an already-fitted model
        """
        X, y = load_concrete(return_dataset=True).to_numpy()

        model = Lasso().fit(X, y)

        with mock.patch.object(model, "fit") as mockfit:
            oz = FeatureImportances(model)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with mock.patch.object(model, "fit") as mockfit:
            oz = FeatureImportances(model, is_fitted=True)
            oz.fit(X, y)
            mockfit.assert_not_called()

        with mock.patch.object(model, "fit") as mockfit:
            oz = FeatureImportances(model, is_fitted=False)
            oz.fit(X, y)
            mockfit.assert_called_once_with(X, y)

    def test_topn_stacked(self):
        """
        Test stack plot with only the three most important features by sum of
        each feature's importance across all classes
        """
        X, y = load_iris(True)

        viz = FeatureImportances(
            LogisticRegression(solver="liblinear", random_state=222),
            stack=True, topn=3
        )
        viz.fit(X, y)
        viz.finalize()

        npt.assert_equal(viz.feature_importances_.shape, (3, 3))
        # Appveyor and Linux conda non-text-based differences
        self.assert_images_similar(viz, tol=17.5)

    def test_topn_negative_stacked(self):
        """
        Test stack plot with only the three least important features by sum of
        each feature's importance across all classes
        """
        X, y = load_iris(True)

        viz = FeatureImportances(
            LogisticRegression(solver="liblinear", random_state=222),
            stack=True, topn=-3
        )
        viz.fit(X, y)
        viz.finalize()

        npt.assert_equal(viz.feature_importances_.shape, (3, 3))
        # Appveyor and Linux conda non-text-based differences
        self.assert_images_similar(viz, tol=17.5)

    def test_topn(self):
        """
        Test plot with only top three important features by absolute value
        """
        X, y = load_iris(True)

        viz = FeatureImportances(
            GradientBoostingClassifier(random_state=42), topn=3
        )
        viz.fit(X, y)
        viz.finalize()

        # Appveyor and Linux conda non-text-based differences
        self.assert_images_similar(viz, tol=17.5)

    def test_topn_negative(self):
        """
        Test plot with only the three least important features by absolute value
        """
        X, y = load_iris(True)

        viz = FeatureImportances(
            GradientBoostingClassifier(random_state=42), topn=-3
        )
        viz.fit(X, y)
        viz.finalize()

        # Appveyor and Linux conda non-text-based differences
        self.assert_images_similar(viz, tol=17.5)


##########################################################################
## Mock Estimator
##########################################################################


class MockEstimator(BaseEstimator):
    """
    Creates params when fit is called on demand.
    """

    def make_importance_param(self, name="feature_importances_", value=None):
        if value is None:
            value = np.random.rand(42)
        setattr(self, name, value)

    def fit(self, X, y=None, **kwargs):
        return self


class MockClassifier(BaseEstimator, ClassifierMixin):
    """
    Creates empty classifier.
    """

    pass
