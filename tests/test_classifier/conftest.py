# tests.test_classifier.conftest
# Provides fixtures for the classification tests module.
#
# Author:  Benjamin Bengfort
# Created: Fri Mar 23 18:07:00 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: conftest.py [1e04216] benjamin@bengfort.com $

"""
Provides fixtures for the classification tests module.
"""

##########################################################################
## Imports
##########################################################################

import pytest

from tests.fixtures import Dataset, Split

from yellowbrick.exceptions import NotFitted
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as tts


##########################################################################
## Assertion Helpers
##########################################################################

ATTRS = ["classes_", "class_count_", "score_"]


def assert_not_fitted(estimator, attrs=ATTRS, X_test=None):
    """
    Check that the estimator is not fitted by ensuring it does not have
    any of the attributes specified in attrs. If X_test is specified,
    it is passed to predict, which must also raise a NotFitted exception.
    """
    __traceback_hide__ = True
    for attr in attrs:
        msg = "model is fitted, has {} attribute".format(attr)
        assert not hasattr(estimator, attr), msg

    if X_test is not None:
        with pytest.raises((NotFitted, NotFittedError)):
            estimator.predict(X_test)


def assert_fitted(estimator, attrs=ATTRS, X_test=None):
    """
    Check that the estimator is fitted by ensuring it does have the attributes
    passed in attrs. If X_test is specified, it is passed to predict which
    must not raise a NotFitted exception.
    """
    __traceback_hide__ = True
    for attr in attrs:
        msg = "model is not fitted, does not have {} attribute".format(attr)
        assert hasattr(estimator, attr), msg

    if X_test is not None:
        try:
            estimator.predict(X_test)
        except (NotFitted, NotFittedError):
            pytest.fail("estimator not fitted raised from predict")


##########################################################################
## Fixtures
##########################################################################


@pytest.fixture(scope="class")
def binary(request):
    """
    Creates a random binary classification dataset fixture
    """
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        n_clusters_per_class=3,
        random_state=87,
    )

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=93)

    dataset = Dataset(Split(X_train, X_test), Split(y_train, y_test))
    request.cls.binary = dataset


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
