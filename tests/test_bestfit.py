# tests.test_bestfit
# Tests for the bestfit module.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Sun Jun 26 19:27:39 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_bestfit.py [56236f3] benjamin@bengfort.com $

"""
Tests for the bestfit module.
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np
import matplotlib.pyplot as plt

from tests.base import VisualTestCase

from yellowbrick.bestfit import *
from yellowbrick.anscombe import ANSCOMBE
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


##########################################################################
## Best fit tests
##########################################################################


class TestBestFit(VisualTestCase):
    def test_bad_estimator(self):
        """
        Test that a bad estimator name raises a value error.
        """
        fig, ax = plt.subplots()
        X, y = ANSCOMBE[1]

        with pytest.raises(YellowbrickValueError):
            draw_best_fit(X, y, ax, "pepper")

    def test_ensure_same_length(self):
        """
        Ensure that vectors of different lengths raise
        """
        fig, ax = plt.subplots()
        X = np.array([1, 2, 3, 5, 8, 10, 2])
        y = np.array([1, 3, 6, 2])

        with pytest.raises(YellowbrickValueError):
            draw_best_fit(X, y, ax, "linear")

        with pytest.raises(YellowbrickValueError):
            draw_best_fit(X[:, np.newaxis], y, ax, "linear")

    @pytest.mark.filterwarnings("ignore")
    def test_draw_best_fit(self):
        """
        Test that drawing a best fit line works.
        """
        fig, ax = plt.subplots()
        X, y = ANSCOMBE[0]

        assert ax == draw_best_fit(X, y, ax, "linear")
        assert ax == draw_best_fit(X, y, ax, "quadratic")


##########################################################################
## Estimator tests
##########################################################################


class TestEstimator(VisualTestCase):
    """
    Test the estimator functions for best fit lines.
    """

    def test_linear(self):
        """
        Test the linear best fit estimator
        """
        X, y = ANSCOMBE[0]
        X = np.array(X)
        y = np.array(y)
        X = X[:, np.newaxis]

        model = fit_linear(X, y)
        assert model is not None
        assert isinstance(model, LinearRegression)

    def test_quadratic(self):
        """
        Test the quadratic best fit estimator
        """
        X, y = ANSCOMBE[1]
        X = np.array(X)
        y = np.array(y)
        X = X[:, np.newaxis]

        model = fit_quadratic(X, y)
        assert model is not None
        assert isinstance(model, Pipeline)

    def test_select_best(self):
        """
        Test the select best fit estimator
        """
        X, y = ANSCOMBE[1]
        X = np.array(X)
        y = np.array(y)
        X = X[:, np.newaxis]

        model = fit_select_best(X, y)
        assert model is not None
        assert isinstance(model, Pipeline)

        X, y = ANSCOMBE[3]
        X = np.array(X)
        y = np.array(y)
        X = X[:, np.newaxis]

        model = fit_select_best(X, y)
        assert model is not None
        assert isinstance(model, LinearRegression)
