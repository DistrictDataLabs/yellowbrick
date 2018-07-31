# tests.test_missing.conftest
# Provides fixtures for the missing tests module.
#
# Author:  Nathan Danielsen <nathan.danielsen@gmail.com>
# Created: Tue July 31 10:42:00 2018 -0400
#
# ID: conftest.py [] nathan.danielsen@gmail.com> $

"""
Provides fixtures for the classification tests module.
"""

##########################################################################
## Imports
##########################################################################

import pytest

import numpy as np
from tests.dataset import Dataset
from sklearn.datasets import make_classification



##########################################################################
## Fixtures
##########################################################################


@pytest.fixture(scope='class')
def missingdata(request):
    """
    Creates a random multiclass classification dataset fixture
    """
    X, y = make_classification(
        n_samples=400, n_features=20, n_informative=8, n_redundant=8,
        n_classes=2, n_clusters_per_class=4, random_state=856
    )

    # add nan values to a range of values in the matrix
    X[X > 1.5] = np.nan
    dataset = Dataset(X, y)
    request.cls.missingdata = dataset
