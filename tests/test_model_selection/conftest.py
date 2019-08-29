# tests.test_model_selection.conftest
# Provides fixtures for the model selection tests module.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Fri Mar 30 14:35:39 2018 -0400
#
# ID: conftest.py [c5355ee] benjamin@bengfort.com $

"""
Provides fixtures for the classification tests module.
"""

##########################################################################
## Imports
##########################################################################

import pytest

from tests.fixtures import Dataset
from sklearn.datasets import make_classification, make_regression, make_blobs


##########################################################################
## Fixtures
##########################################################################


@pytest.fixture(scope="class")
def classification(request):
    """
    Creates a random multiclass classification dataset fixture
    """
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=3,
        random_state=3902,
    )

    dataset = Dataset(X, y)
    request.cls.classification = dataset


@pytest.fixture(scope="class")
def regression(request):
    """
    Creates a random regression dataset fixture
    """
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=8,
        noise=0.01,
        bias=1.4,
        random_state=953,
    )

    dataset = Dataset(X, y)
    request.cls.regression = dataset


@pytest.fixture(scope="class")
def clusters(request):
    """
    Creates a random regression dataset fixture
    """
    X, y = make_blobs(n_samples=500, n_features=20, centers=3, random_state=743)

    dataset = Dataset(X, y)
    request.cls.clusters = dataset
