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
from yellowbrick.regressor import residuals_plot
from yellowbrick.classifier import classification_report
from yellowbrick.exceptions import YellowbrickAttributeError
from yellowbrick.utils.helpers import get_model_name, is_fitted
from yellowbrick.utils.types import is_estimator, is_probabilistic
from yellowbrick.utils.types import is_classifier, is_clusterer, is_regressor

from sklearn.base import is_regressor as sk_is_regressor
from sklearn.base import is_classifier as sk_is_classifier
from sklearn.model_selection import train_test_split as tts
from sklearn.datasets import make_classification, make_regression
from sklearn.base import is_outlier_detector as sk_is_outlier_detector


try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import cudf
    from cuml.ensemble import RandomForestClassifier as curfc
except ImportError:
    curfc = None

try:
    import catboost
except ImportError:
    catboost = None


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


##########################################################################
## Test Non-sklearn Estimators
##########################################################################

class TestNonSklearnEstimators(object):
    """
    Check various non-sklearn estimators to see if the wrapper works for them
    """

    @pytest.mark.skipif(xgb is None, reason="requires xgboost")
    def test_xgboost_regressor(self):
        """
        Validate xgboost regressor with wrapper
        """
        X, y = make_regression(
            n_samples=500, n_features=22, n_informative=8, random_state=8311982
        )
        X_train, X_test, y_train, y_test = tts(X, y)

        model = regressor(xgb.XGBRFRegressor())
        oz = residuals_plot(model, X_train, y_train, X_test, y_test, show=False)
        assert is_fitted(oz)


    @pytest.mark.skipif(xgb is None, reason="requires xgboost")
    def test_xgboost_regressor_unwrapped(self):
        """
        Validate xgboost regressor without wrapper
        """
        X, y = make_regression(
            n_samples=500, n_features=22, n_informative=8, random_state=8311982
        )
        X_train, X_test, y_train, y_test = tts(X, y)

        model = xgb.XGBRFRegressor()
        oz = residuals_plot(model, X_train, y_train, X_test, y_test, show=False)
        assert is_fitted(oz)

    @pytest.mark.skipif(curfc is None, reason="requires cuML")
    def test_cuml_classifier(self):
        """
        Validate cuML classifier with wrapper
        """
        # NOTE: this is currently untested as I wasn't able to install cuML
        X, y = make_classification(
            n_samples=400, n_features=10, n_informative=2, n_redundant=3,
            n_classes=2, n_clusters_per_class=2, random_state=8311982
        )
        X_train, X_test, y_train, y_test = tts(X, y)

        # Convert to cudf dataframes
        X_train = cudf.DataFrame(X_train)
        y_train = cudf.Series(y_train)
        X_test = cudf.DataFrame(X_test)
        y_test = cudf.Series(y_test)

        model = classifier(curfc(n_estimators=40, max_depth=8, max_features=1))
        oz = classification_report(model, X_train, y_train, X_test, y_test, show=False)
        assert is_fitted(oz)

    @pytest.mark.skipif(catboost is None, reason="requires CatBoost")
    def test_catboost_classifier(self):
        """
        Validate CatBoost classifier with wrapper
        """
        X, y = make_classification(
            n_samples=400, n_features=10, n_informative=2, n_redundant=3,
            n_classes=2, n_clusters_per_class=2, random_state=8311982
        )
        X_train, X_test, y_train, y_test = tts(X, y)

        model = classifier(catboost.CatBoostClassifier(
            iterations=2, depth=2, learning_rate=1, loss_function='Logloss'
        ))

        # For some reason, this works if you call fit directly and pass is_fitted to
        # the visualizer, but does not work if you rely on the visualizer to fit the
        # model on the data. I can't tell if this is a catboost or Yellowbrick issue.
        model.fit(X_train, y_train)

        oz = classification_report(
            model, X_train, y_train, X_test, y_test, is_fitted=True, show=False
        )
        assert is_fitted(oz)
