# tests.test_classifier.test_base
# Tests for the base classification visualizers
#
# Author:   Benjamin Bengfort
# Created:  Wed Jul 31 11:21:28 2019 -0400
#
# Copyright (C) 2019 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_base.py [] benjamin@bengfort.com $

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

    def test_colors_property(self):
        """
        Test that a unique color per class is created after fit
        """
        oz = ClassificationScoreVisualizer(GaussianNB())

        with pytest.raises(NotFitted, match="cannot determine colors before fit"):
            oz.colors

        oz.fit(self.multiclass.X.train, self.multiclass.y.train)
        assert len(oz.colors) == len(oz.classes_)

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
