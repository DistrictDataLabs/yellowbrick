# tests.test_features.test_base
# Tests for the feature selection and analysis base classes
#
# Author:   Benjamin Bengfort
# Created:  Fri Oct 07 13:43:55 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_base.py [2e898a6] benjamin@bengfort.com $

"""
Tests for the feature selection and analysis base classes
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np
import numpy.testing as npt

from tests.fixtures import Dataset
from yellowbrick.base import Visualizer
from yellowbrick.features.base import *

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_classification, make_regression

from unittest.mock import Mock

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Fixtures
##########################################################################


@pytest.fixture(scope="class")
def discrete(request):
    """
    Create a random classification dataset fixture.
    """
    X, y = make_classification(
        n_classes=5,
        n_samples=400,
        n_features=12,
        n_informative=10,
        n_redundant=0,
        random_state=2019,
    )

    # Dataset is accessible on the class so it is only generated once
    request.cls.discrete = Dataset(X, y)


@pytest.fixture(scope="class")
def continuous(request):
    """
    Creates a random regression dataset fixture.
    """
    X, y = make_regression(
        n_samples=500, n_features=22, n_informative=8, random_state=2019
    )

    # Dataset is accessible on the class so it is only generated once
    request.cls.continuous = Dataset(X, y)


##########################################################################
## FeatureVisualizer Base Tests
##########################################################################


@pytest.mark.usefixtures("discrete")
class TestFeatureVisualizer(object):
    """
    Test FeatureVisualizer base class
    """

    def test_subclass(self):
        """
        Check the visualizer/transformer class hierarchy
        """
        visualizer = FeatureVisualizer()
        assert isinstance(visualizer, TransformerMixin)
        assert isinstance(visualizer, BaseEstimator)
        assert isinstance(visualizer, Visualizer)

    def test_transform_returns_unmodified(self):
        """
        Ensure transformer is just a passthrough
        """
        X, y = self.discrete
        visualizer = FeatureVisualizer().fit(X, y)
        assert visualizer.transform(X, y) is X

    def test_fit_transform_show(self):
        """
        Test the fit/transform/show quick method
        """

        class MockFeatureVisaulizer(FeatureVisualizer):
            pass

        viz = MockFeatureVisaulizer()
        viz.fit = Mock(return_value=viz)
        viz.transform = Mock(return_value="a")
        viz.show = Mock()

        X, y = self.discrete
        assert viz.fit_transform_show(X, y, outpath="a.png", clear_figure=True) == "a"
        assert viz.fit.called_once_with(X, y)
        assert viz.transform.called_once_with(X, y)
        assert viz.show.called_once_with(outpath="a.png", clear_figure=True)


##########################################################################
## MultiFeatureVisualizer Tests
##########################################################################


@pytest.mark.usefixtures("discrete")
class TestMultiFeatureVisualizer(object):
    """
    Test the MultiFeatureVisualizer base class
    """

    def test_subclass(self):
        """
        Check the visualizer/transformer class hierarchy
        """
        visualizer = MultiFeatureVisualizer()
        assert isinstance(visualizer, FeatureVisualizer)
        assert isinstance(visualizer, TransformerMixin)
        assert isinstance(visualizer, BaseEstimator)
        assert isinstance(visualizer, Visualizer)

    def test_user_supplied_features(self):
        """
        Test that a user can supply feature names directly
        """
        X, y = self.discrete
        features = ["f{}".format(i + 1) for i in range(X.shape[1])]
        oz = MultiFeatureVisualizer(features=features)

        assert not hasattr(oz, "features_")
        assert oz.fit(X, y) is oz
        assert hasattr(oz, "features_")
        npt.assert_array_equal(oz.features_, np.asarray(features))

    def test_numeric_features(self):
        """
        Test that the features are column indices for numpy arrays
        """
        X, y = self.discrete
        oz = MultiFeatureVisualizer()

        assert not hasattr(oz, "features_")
        assert oz.fit(X, y) is oz
        assert hasattr(oz, "features_")
        assert len(oz.features_) == X.shape[1]

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_string_columns(self):
        """
        Ensure that DataFrame column names are uses as features
        """
        X, y = self.discrete
        features = ["f{}".format(i + 1) for i in range(X.shape[1])]

        X = pd.DataFrame(X, columns=features)
        y = pd.Series(y, name="target")

        oz = MultiFeatureVisualizer()
        assert not hasattr(oz, "features_")
        assert oz.fit(X, y) is oz
        assert hasattr(oz, "features_")
        npt.assert_array_equal(oz.features_, np.asarray(features))

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_pandas_no_columns(self):
        """
        Test when a DataFrame has no column names
        """
        """
        Ensure that Pandas column names are uses as features
        """
        X, y = self.discrete

        X = pd.DataFrame(X)
        y = pd.Series(y, name="target")

        oz = MultiFeatureVisualizer()
        assert not hasattr(oz, "features_")
        assert oz.fit(X, y) is oz
        assert hasattr(oz, "features_")
        assert len(oz.features_) == X.shape[1]


##########################################################################
## DataVisualizer Tests
##########################################################################


@pytest.mark.usefixtures("discrete", "continuous")
class TestDataVisualizer(object):
    """
    Test DataVisualizer base class
    """

    FIELDS = (
        "features_",
        "classes_",
        "range_",
        "_colors",
        "_target_color_type",
        "_label_encoder",
    )

    def assert_not_fitted(self, obj):
        __tracebackhide__ = True
        for field in self.FIELDS:
            if field.endswith("_"):
                assert not hasattr(obj, field), "has {} before fit".format(field)
            else:
                msg = "missing internal var {}".format(field)
                assert getattr(obj, field) is None, msg

    def assert_fitted(self, obj, fields=FIELDS):
        __tracebackhide__ = True
        # Mutually exclusive fields
        if obj._target_color_type == TargetType.SINGLE:
            assert (not hasattr(obj, "classes_")) and (not hasattr(obj, "range_"))

        elif obj._target_color_type == TargetType.DISCRETE:
            assert hasattr(obj, "_label_encoder")
            assert hasattr(obj, "classes_") and (not hasattr(obj, "range_"))

        elif obj._target_color_type == TargetType.CONTINUOUS:
            assert (not hasattr(obj, "classes_")) and hasattr(obj, "range_")

        else:
            raise ValueError(
                "cannot test target type {}".format(obj._target_color_type)
            )

        for field in fields:
            if field in {"classes_", "range_", "_label_encoder"}:
                continue  # handled by mutually exclusive
            assert hasattr(obj, field)

    def test_single_when_none(self):
        """
        Ensure that the target type is "single" when y is None
        """
        X, _ = self.discrete
        oz = DataVisualizer()
        assert oz.target_type == TargetType.AUTO

        # Assert single when y is None
        self.assert_not_fitted(oz)
        assert oz.fit(X, y=None) is oz
        self.assert_fitted(oz)
        assert oz._target_color_type == TargetType.SINGLE

    @pytest.mark.parametrize("target_type", ("discrete", "continuous"))
    def test_none_overrides_user_specified_target(self, target_type):
        """
        Even if a user supplies a target type it should be overriden by y=None
        """
        X, _ = getattr(self, target_type)
        oz = DataVisualizer(target_type=target_type).fit(X, y=None)

        assert oz._colors == "C0"
        assert oz._target_color_type == TargetType.SINGLE

    @pytest.mark.parametrize("dataset", ("discrete", "continuous"))
    def test_user_overrides_auto_target(self, dataset):
        """
        Ensure user specified target type overrides auto discovery
        """
        X, y = getattr(self, dataset)
        target_type = (
            TargetType.CONTINUOUS if dataset == "discrete" else TargetType.DISCRETE
        )

        oz = DataVisualizer(target_type=target_type)
        assert oz.target_type != TargetType.AUTO
        oz.fit(X, y)
        assert oz.target_type == target_type

    def test_continuous(self):
        """
        Test data visualizer on continuous data
        """
        # Check when y is continuous
        X, y = self.continuous
        oz = DataVisualizer()
        assert oz.target_type == TargetType.AUTO

        self.assert_not_fitted(oz)
        assert oz.fit(X, y) is oz
        self.assert_fitted(oz)
        assert oz._target_color_type == TargetType.CONTINUOUS
        assert oz.range_ == (y.min(), y.max())

    def test_discrete(self):
        """
        Test data visualizer on discrete data
        """
        # Check when y is discrete
        X, y = self.discrete
        oz = DataVisualizer()
        assert oz.target_type == TargetType.AUTO

        self.assert_not_fitted(oz)
        assert oz.fit(X, y) is oz
        self.assert_fitted(oz)
        assert oz._target_color_type == TargetType.DISCRETE
        assert len(oz.classes_) == np.unique(y).shape[0]

    def test_bad_target_type(self):
        """
        Assert target type is validated on init
        """
        msg = "unknown target color type 'foo'"
        with pytest.raises(YellowbrickValueError, match=msg):
            DataVisualizer(target_type="foo")

    def test_classes_discrete(self):
        """
        Ensure classes are assigned correctly for label encoding
        """
        X, y = self.discrete
        classes = ["a", "b", "c", "d", "e"]
        oz = DataVisualizer(classes=classes, target_type="discrete").fit(X, y)

        npt.assert_array_equal(oz.classes_, classes)
        assert list(oz._colors.keys()) == classes

    def test_classes_continuous(self):
        """
        Ensure classes are ignored in continuous case
        """
        X, y = self.continuous
        classes = ["a", "b", "c", "d", "e"]
        oz = DataVisualizer(classes=classes, target_type="continuous").fit(X, y)

        assert not hasattr(oz, "classes_")

    def test_get_target_color_type(self):
        """
        Test the get_target_color_type helper method
        """
        oz = DataVisualizer()

        with pytest.raises(NotFitted, match="unknown target color type"):
            oz.get_target_color_type()

        oz.fit(*self.continuous)
        assert oz.get_target_color_type() == TargetType.CONTINUOUS

    def test_get_colors_not_fitted(self):
        """
        Assert get_colors requires an fitted visualizer
        """
        oz = DataVisualizer()
        with pytest.raises(NotFitted, match="cannot determine colors"):
            oz.get_colors(["a", "b", "c"])

    @pytest.mark.parametrize("color, expected", [(None, "C0"), ("#F3B8AB", "#F3B8AB")])
    def test_get_colors_single(self, color, expected):
        """
        Test color assignment for single target type
        """
        X, y = self.discrete
        oz = DataVisualizer(colors=color).fit(X)
        assert oz.get_target_color_type() == TargetType.SINGLE

        # Test default colors
        colors = oz.get_colors(y)
        assert len(colors) == len(y)
        assert np.unique(colors) == expected

    def test_get_colors_discrete(self):
        """
        Test discrete colors with no label encoding
        """
        X, y = self.discrete
        oz = DataVisualizer().fit(X, y)
        assert oz.get_target_color_type() == TargetType.DISCRETE

        colors = oz.get_colors(y)
        assert len(colors) == len(y)
        assert set(colors) == set(oz._colors.values())

    def test_get_colors_discrete_classes(self):
        """
        Test discrete colors with label encoding and colors
        """
        X, y = self.discrete
        oz = DataVisualizer(
            classes=["a", "b", "c", "d", "e"], colors=["g", "r", "b", "m", "y"]
        ).fit(X, y)
        assert oz.get_target_color_type() == TargetType.DISCRETE

        colors = oz.get_colors(y)
        assert len(colors) == len(y)
        assert set(colors) == set(["g", "r", "b", "m", "y"])

    def test_get_colors_not_label_encoded(self):
        """
        Assert exception is raised on unknown class label for get_colors
        """
        X, y = self.discrete
        oz = DataVisualizer(classes="abcde").fit(X, y)

        with pytest.raises(YellowbrickKeyError, match="could not determine color"):
            oz.get_colors(["foo"])

    @pytest.mark.parametrize(
        "colors, colormap",
        [
            (["#3f78de", "#f38b33"], None),
            (["b", "g", "r", "m", "y"], None),
            (None, "Blues"),
        ],
    )
    def test_user_get_colors_discrete(self, colors, colormap):
        """
        Test the ways that users can specify colors
        """
        X, y = self.discrete
        oz = DataVisualizer(
            colors=colors, colormap=colormap, target_type="discrete"
        ).fit(X, y)

        colors = oz.get_colors(y)
        assert len(colors) == len(y)

    def test_get_colors_continous(self):
        """
        Test continuous colors with no default colormap
        """
        X, y = self.continuous
        oz = DataVisualizer().fit(X, y)
        assert oz.get_target_color_type() == TargetType.CONTINUOUS

        colors = oz.get_colors(y)
        assert len(colors) == len(y)

    def test_get_colors_continous_cmap(self):
        """
        Test continuous colors with user specified cmap
        """
        X, y = self.continuous
        oz = DataVisualizer(colormap="jet").fit(X, y)
        assert oz.get_target_color_type() == TargetType.CONTINUOUS

        colors = oz.get_colors(y)
        assert len(colors) == len(y)
