# tests.test_features.conftest
# Provides fixtures for the feature tests module.
#
# Author:   Naresh Bachwani
# Created:  Thu Aug 15 07:35:53 2019 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: conftest.py [2c5f0e9] 43993586+naresh-bachwani@users.noreply.github.com $

"""
Provides fixtures for the feature tests module.
"""

##########################################################################
## Imports
##########################################################################

import pytest

from tests.fixtures import Dataset
from sklearn.datasets import make_classification, make_regression, make_s_curve


##########################################################################
## Fixtures
##########################################################################


@pytest.fixture(scope="class")
def discrete(request):
    """
    Creates a fixture of train and test splits for the sklearn digits dataset
    For ease of use returns a Dataset named tuple composed of two Split tuples.
    """
    X, y = make_classification(
        n_samples=400,
        n_features=12,
        n_informative=8,
        n_redundant=0,
        n_classes=5,
        n_clusters_per_class=1,
        class_sep=1.8,
        random_state=854,
        scale=[14.2, 2.1, 0.32, 0.001, 32.3, 44.1, 102.3, 2.3, 2.4, 38.2, 0.05, 1.0],
    )

    # Set a class attribute for discrete data.
    request.cls.discrete = Dataset(X, y)


@pytest.fixture(scope="class")
def continuous(request):
    """
    Creates a random regressor fixture.
    """
    X, y = make_regression(
        n_samples=500, n_features=22, n_informative=8, random_state=2019
    )

    # Set a class attribute for continuous data
    request.cls.continuous = Dataset(X, y)


@pytest.fixture(scope="class")
def s_curves(request):
    """
    Creates a random regressor fixture.
    """
    X, y = make_s_curve(1000, random_state=888)
    # Set a class attribute for continuous data
    request.cls.s_curves = Dataset(X, y)
