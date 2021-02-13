# tests.test_contrib.test_missing.test_dispersion
# Tests for the alpha selection visualizations.
#
# Author:  Nathan Danielsen <nathan.danielsen@gmail.com>
# Created:  Thu Mar 29 12:13:04 2018 -0500
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_dispersion.py [1443e16] ndanielsen@users.noreply.github.com $

"""
Tests for the MissingValuesDispersion visualizations.
"""

##########################################################################
## Imports
##########################################################################

import os
import pytest

from sklearn.datasets import make_classification
from tests.base import VisualTestCase

from yellowbrick.contrib.missing.dispersion import *

try:
    import pandas as pd
except ImportError:
    pd = None


@pytest.fixture(scope="class")
def missing_dispersion_tolerance(request):
    request.cls.tol = 0.5 if os.name == "nt" else 0.01


##########################################################################
## Feature Importances Tests
##########################################################################


@pytest.mark.usefixtures("missing_dispersion_tolerance")
class TestMissingValuesDispersion(VisualTestCase):
    """
    MissingValuesDispersion visualizer
    """

    @pytest.mark.skipif(pd is None, reason="pandas is required")
    def test_missingvaluesdispersion_with_pandas(self):
        """
        Integration test of visualizer with pandas
        """
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=8,
            n_redundant=8,
            n_classes=2,
            n_clusters_per_class=4,
            random_state=854,
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        X_ = pd.DataFrame(X)
        features = [str(n) for n in range(20)]
        viz = MissingValuesDispersion(features=features)
        viz.fit(X_)
        viz.finalize()

        self.assert_images_similar(viz, tol=self.tol)

    @pytest.mark.skipif(pd is None, reason="pandas is required")
    def test_missingvaluesdispersion_with_pandas_with_y_targets(self):
        """
        Integration test of visualizer with pandas with y targets
        """
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=8,
            n_redundant=8,
            n_classes=2,
            n_clusters_per_class=4,
            random_state=854,
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        X_ = pd.DataFrame(X)
        features = [str(n) for n in range(20)]
        classes = ["Class A", "Class B"]
        viz = MissingValuesDispersion(features=features, classes=classes)
        viz.fit(X_, y=y)
        viz.finalize()

        self.assert_images_similar(viz, tol=self.tol)

    def test_missingvaluesdispersion_with_numpy(self):
        """
        Integration test of visualizer with numpy
        """
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=8,
            n_redundant=8,
            n_classes=2,
            n_clusters_per_class=4,
            random_state=852,
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        features = [str(n) for n in range(20)]
        viz = MissingValuesDispersion(features=features)
        viz.fit(X)
        viz.finalize()

        self.assert_images_similar(viz, tol=self.tol)

    def test_missingvaluesdispersion_with_numpy_with_y_targets(self):
        """
        Integration test of visualizer with numpy with y targets
        """
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=8,
            n_redundant=8,
            n_classes=2,
            n_clusters_per_class=4,
            random_state=852,
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        features = [str(n) for n in range(20)]
        classes = ["Class A", "Class B"]
        viz = MissingValuesDispersion(features=features, classes=classes)
        viz.fit(X, y=y)
        viz.finalize()

        self.assert_images_similar(viz, tol=self.tol)
