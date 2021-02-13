# tests.test_classifier.test_base
# Tests for the base classification visualizers
#
# Author:   Benjamin Bengfort
# Created:  Wed Jul 31 11:21:28 2019 -0400
#
# Copyright (C) 2019 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_base.py [da729da] benjamin@bengfort.com $

"""
Tests for the base classification visualizers
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np
import numpy.testing as npt

from yellowbrick.classifier.base import *
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from .conftest import assert_fitted, assert_not_fitted
from yellowbrick.exceptions import YellowbrickTypeError


##########################################################################
## Test Classification Score Visualizer
##########################################################################


@pytest.mark.usefixtures("binary", "multiclass")
class TestClassificationScoreVisualizer(object):
    """
    Test the ClassificationScoreVisualizer base functionality
    """

    def test_fit_score(self):
        """
        Ensure correct fit and score behavior
        """
        oz = ClassificationScoreVisualizer(GaussianNB())
        assert_not_fitted(oz, X_test=self.binary.X.test)
        assert oz.fit(self.binary.X.train, self.binary.y.train) is oz
        assert 0.0 <= oz.score(self.binary.X.test, self.binary.y.test) <= 1.0
        assert_fitted(oz, X_test=self.binary.X.test)

    def test_class_counts(self):
        """
        Test class and class counts identification
        """
        oz = ClassificationScoreVisualizer(GaussianNB())
        oz.fit(self.multiclass.X.train, self.multiclass.y.train)

        unique, counts = np.unique(self.multiclass.y.train, return_counts=True)
        npt.assert_array_equal(oz.classes_, unique)
        npt.assert_array_equal(oz.class_counts_, counts)

    def test_force_estimator(self):
        """
        Test that an estimator can be forced through
        """
        with pytest.raises(YellowbrickTypeError):
            ClassificationScoreVisualizer(LinearRegression())

        try:
            ClassificationScoreVisualizer(LinearRegression(), force_model=True)
        except YellowbrickTypeError as e:
            pytest.fail("type error was raised incorrectly: {}".format(e))

    def test_score_with_fitted_estimator(self):
        """
        Assert fitted estimator can be scored without fit but warns
        """
        model = GaussianNB().fit(self.binary.X.train, self.binary.y.train)

        # NOTE that the wrapper will pass a call down to `classes_`
        oz = ClassificationScoreVisualizer(model)
        assert_not_fitted(oz, ["class_counts_", "score_"])

        msg = "could not determine class_counts_"
        with pytest.warns(YellowbrickWarning, match=msg):
            oz.score(self.binary.X.test, self.binary.y.test)
            assert_fitted(oz, ["classes_", "class_counts_", "score_"])

    def test_score_without_fitted_estimator(self):
        """
        Assert score without fitted estimator raises NotFitted
        """
        oz = ClassificationScoreVisualizer(GaussianNB())
        assert_not_fitted(oz)

        with pytest.raises(NotFitted):
            oz.score(self.binary.X.test, self.binary.y.test)
            assert_not_fitted(oz)

    def test_colors_property(self):
        """
        Test that a unique color per class is created after fit
        """
        oz = ClassificationScoreVisualizer(GaussianNB())

        with pytest.raises(NotFitted, match="cannot determine colors before fit"):
            oz.class_colors_

        oz.fit(self.multiclass.X.train, self.multiclass.y.train)
        assert len(oz.class_colors_) == len(oz.classes_)

    def test_decode_labels_warning(self):
        """
        Assert warning is issued and encoder is used with multiple decoding params
        """
        with pytest.warns(
            YellowbrickWarning, match="both classes and encoder specified"
        ):
            oz = ClassificationScoreVisualizer(
                GaussianNB(),
                classes=["a", "b", "c"],
                encoder={0: "foo", 1: "bar", 2: "zap"},
            )
            encoded = oz._decode_labels([0, 1, 2])
            npt.assert_array_equal(encoded, ["foo", "bar", "zap"])

    def test_decode_labels_from_numeric(self):
        """
        Test that a numeric y can be decoded using classes and encoder
        """
        classes = np.array(["a", "b", "c", "d", "e"])
        y = np.random.randint(0, 5, 100)
        decoded = classes[y]

        oz = ClassificationScoreVisualizer(GaussianNB, classes=classes)
        npt.assert_array_equal(oz._decode_labels(y), decoded)

        encoder = dict(zip(range(len(classes)), classes))
        oz = ClassificationScoreVisualizer(GaussianNB, encoder=encoder)
        npt.assert_array_equal(oz._decode_labels(y), decoded)

        encoder = LabelEncoder().fit(decoded)
        oz = ClassificationScoreVisualizer(GaussianNB, encoder=encoder)
        npt.assert_array_equal(oz._decode_labels(y), decoded)

    def test_decode_labels_from_strings(self):
        """
        Test that string y can be decoded using classes and encoder
        """
        classes = np.array(["a", "b", "c", "d", "e"])
        decoded = classes[np.random.randint(0, 5, 100)]
        y = np.array([v.upper() for v in decoded])

        oz = ClassificationScoreVisualizer(GaussianNB, classes=classes)
        npt.assert_array_equal(oz._decode_labels(y), decoded)

        encoder = {c.upper(): c for c in classes}
        oz = ClassificationScoreVisualizer(GaussianNB, encoder=encoder)
        npt.assert_array_equal(oz._decode_labels(y), decoded)

        class L2UTransformer(object):
            def transform(self, y):
                return np.array([yi.upper() for yi in y])

            def inverse_transform(self, y):
                return np.array([yi.lower() for yi in y])

        oz = ClassificationScoreVisualizer(GaussianNB, encoder=L2UTransformer())
        npt.assert_array_equal(oz._decode_labels(y), decoded)

    def test_decode_labels_unknown_class(self):
        """
        Ensure a human-understandable error is raised when decode fails
        """
        classes = np.array(["a", "b", "c", "d", "e"])
        y = classes[np.random.randint(0, 5, 100)]

        # Remove class "c" from the known array labels
        classes = np.array(["a", "b", "d", "e"])

        oz = ClassificationScoreVisualizer(GaussianNB, classes=classes)
        with pytest.raises(ModelError, match="could not decode"):
            npt.assert_array_equal(oz._decode_labels(y), decoded)

        encoder = dict(zip(classes, range(len(classes))))
        oz = ClassificationScoreVisualizer(GaussianNB, encoder=encoder)
        with pytest.raises(ModelError, match="cannot decode class 'c' to label"):
            npt.assert_array_equal(oz._decode_labels(y), decoded)

        encoder = LabelEncoder().fit(classes[np.random.randint(0, 4, 100)])
        oz = ClassificationScoreVisualizer(GaussianNB, encoder=encoder)
        with pytest.raises(ModelError, match="could not decode"):
            npt.assert_array_equal(oz._decode_labels(y), decoded)

    def test_labels(self):
        """
        Check visualizer can return human labels correctly
        """
        classes = np.array(["a", "b", "c", "d", "e"])
        y = classes[np.random.randint(0, 5, 100)]

        oz = ClassificationScoreVisualizer(GaussianNB, classes=classes)
        npt.assert_array_equal(oz._labels(), classes)

        encoder = dict(zip(range(len(classes)), classes))
        oz = ClassificationScoreVisualizer(GaussianNB, encoder=encoder)
        npt.assert_array_equal(oz._labels(), classes)

        encoder = LabelEncoder().fit(y)
        oz = ClassificationScoreVisualizer(GaussianNB, encoder=encoder)
        npt.assert_array_equal(oz._labels(), classes)

    def test_labels_warning(self):
        """
        Assert warning and encoder is used with multiple decoding params for labels
        """
        with pytest.warns(
            YellowbrickWarning, match="both classes and encoder specified"
        ):
            oz = ClassificationScoreVisualizer(
                GaussianNB(),
                classes=["a", "b", "c"],
                encoder={0: "foo", 1: "bar", 2: "zap"},
            )
            labels = oz._labels()
            npt.assert_array_equal(labels, ["foo", "bar", "zap"])

    def test_labels_encoder_no_classes(self):
        """
        Assert warning and None returned if encoder doesn't have classes
        """

        class L2UTransformer(object):
            def transform(self, y):
                return np.array([yi.upper() for yi in y])

        oz = ClassificationScoreVisualizer(GaussianNB(), encoder=L2UTransformer())
        with pytest.warns(YellowbrickWarning, match="could not determine class labels"):
            assert oz._labels() is None

    def test_dict_labels_sorted(self):
        """
        Ensure dictionary encoder labels are returned sorted
        """
        le = {3: "a", 2: "c", 1: "b"}
        oz = ClassificationScoreVisualizer(GaussianNB(), encoder=le)
        npt.assert_array_equal(oz._labels(), ["b", "c", "a"])
