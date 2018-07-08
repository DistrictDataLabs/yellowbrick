# tests.test_classifier.conftest
# Provides fixtures for the classification tests module.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Fri Mar 23 18:07:00 2018 -0400
#
# ID: conftest.py [] benjamin@bengfort.com $

"""
Provides fixtures for the classification tests module.
"""

##########################################################################
## Imports
##########################################################################

import pytest

from tests.dataset import Dataset, Split

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as tts


##########################################################################
## Fixtures
##########################################################################

@pytest.fixture(scope='class')
def binary(request):
    """
    Creates a random binary classification dataset fixture
    """
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=8, n_redundant=2,
        n_classes=2, n_clusters_per_class=3, random_state=87
    )

    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.2, random_state=93
    )

    dataset = Dataset(Split(X_train, X_test), Split(y_train, y_test))
    request.cls.binary = dataset


@pytest.fixture(scope='class')
def multiclass(request):
    """
    Creates a random multiclass classification dataset fixture
    """
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=8, n_redundant=2,
        n_classes=6, n_clusters_per_class=3, random_state=87
    )

    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.2, random_state=93
    )

    dataset = Dataset(Split(X_train, X_test), Split(y_train, y_test))
    request.cls.multiclass = dataset
