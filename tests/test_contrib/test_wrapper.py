# tests.test_contrib.test_wrapper
# Tests third-party estimator wrapper
#
# Author:   Benjamin Bengfort
# Created:  Fri Oct 02 17:27:50 2020 -0400
#
# Copyright (C) 2020 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_wrapper.py [] benjamin@bengfort.com $

"""
Tests third-party estimator wrapper
"""

##########################################################################
## Imports
##########################################################################

import pytest


from yellowbrick.contrib.wrapper import *
from yellowbrick.utils.helpers import get_model_name
from yellowbrick.exceptions import YellowbrickAttributeError
from yellowbrick.utils.types import is_estimator, is_probabilistic
from yellowbrick.utils.types import is_classifier, is_clusterer, is_regressor

from sklearn.base import is_regressor as sk_is_regressor
from sklearn.base import is_classifier as sk_is_classifier
from sklearn.base import is_outlier_detector as sk_is_outlier_detector


##########################################################################
## Mocks and Fixtures
##########################################################################

class ThirdPartyEstimator(object):

    def __init__(self, **params):
        for attr, param in params.items():
            setattr(self, attr, param)

    def fit(self, X, y=None):
        return 42

    def predict_proba(self, X, y=None):
        return 24


##########################################################################
## Test Suite
##########################################################################

class TestContribWrapper(object):
    """
    Third party ContribEstimator wrapper
    """

    def test_wrapped_estimator(self):
        """
        Check that the contrib wrapper passes through correctly
        """
        tpe = ContribEstimator(ThirdPartyEstimator(foo="bar"), "foo")
        assert tpe.fit([1,2,3]) == 42
        assert tpe._estimator_type == "foo"

    def test_attribute_error(self):
        """
        Assert a correct exception is raised on failed access
        """
        tpe = ContribEstimator(ThirdPartyEstimator())
        with pytest.raises(YellowbrickAttributeError):
            tpe.foo

    def test_get_model_name(self):
        """
        ContribWrapper should return the underlying model name
        """
        tpe = ContribEstimator(ThirdPartyEstimator())
        assert get_model_name(tpe) == "ThirdPartyEstimator"

    def test_wraps_is_estimator(self):
        """
        Assert a wrapped estimator passes is_estimator check
        """
        tpe = wrap(ThirdPartyEstimator())
        assert is_estimator(tpe)

    def test_wraps_is_classifier(self):
        """
        Assert a wrapped estimator passes is_classifier check
        """
        tpe = classifier(ThirdPartyEstimator())
        assert is_classifier(tpe)
        assert sk_is_classifier(tpe)

    def test_wraps_is_regressor(self):
        """
        Assert a wrapped estimator passes is_regressor check
        """
        tpe = regressor(ThirdPartyEstimator())
        assert is_regressor(tpe)
        assert sk_is_regressor(tpe)

    def test_wraps_is_clusterer(self):
        """
        Assert a wrapped estimator passes is_clusterer check
        """
        tpe = clusterer(ThirdPartyEstimator())
        assert is_clusterer(tpe)

    def test_wraps_is_outlier_detector(self):
        """
        Assert a wrapped estimator passes is_outlier_detector check
        """
        tpe = wrap(ThirdPartyEstimator(), OUTLIER_DETECTOR)
        assert sk_is_outlier_detector(tpe)

    def test_wraps_is_probabilistic(self):
        """
        Assert a wrapped estimator passes is_probabilistic check
        """
        tpe = wrap(ThirdPartyEstimator())
        assert is_probabilistic(tpe)
