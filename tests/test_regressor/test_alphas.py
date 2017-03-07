# tests.test_regressor.test_alphas
# Tests for the alpha selection visualizations.
#
# Author:   Benjamin Bengfort <benjamin@bengfort.com>
# Created:  codetime
#
# Copyright (C) 2016 Bengfort.com
# For license information, see LICENSE.txt
#
# ID: tests.test_regressor.test_alphas.py [] benjamin@bengfort.com $

"""
Tests for the alpha selection visualizations.
"""

##########################################################################
## Imports
##########################################################################

import unittest
import numpy as np

from tests.base import VisualTestCase
from yellowbrick.regressor.alphas import *
from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LassoLars, LassoLarsCV
from sklearn.linear_model import ElasticNet, ElasticNetCV


##########################################################################
## Data
##########################################################################

X = np.array(
        [[ 2.318, 2.727, 4.260, 7.212, 4.792],
         [ 2.315, 2.726, 4.295, 7.140, 4.783,],
         [ 2.315, 2.724, 4.260, 7.135, 4.779,],
         [ 2.110, 3.609, 4.330, 7.985, 5.595,],
         [ 2.110, 3.626, 4.330, 8.203, 5.621,],
         [ 2.110, 3.620, 4.470, 8.210, 5.612,],
         [ 2.318, 2.727, 4.260, 7.212, 4.792,],
         [ 2.315, 2.726, 4.295, 7.140, 4.783,],
         [ 2.315, 2.724, 4.260, 7.135, 4.779,],
         [ 2.110, 3.609, 4.330, 7.985, 5.595,],
         [ 2.110, 3.626, 4.330, 8.203, 5.621,],
         [ 2.110, 3.620, 4.470, 8.210, 5.612,]]
    )

y = np.array([0.23, .33, .31, .3, .24, .32, 0.23, .33, .31, .3, .24, .32])


##########################################################################
## Alpha Selection Tests
##########################################################################

class AlphaSelectionTests(VisualTestCase):

    def test_regressor_cv(self):
        """
        Ensure only "CV" regressors are allowed
        """

        for model in (SVR, Ridge, Lasso, LassoLars, ElasticNet):
            with self.assertRaises(YellowbrickTypeError):
                alphas = AlphaSelection(model())

        for model in (RidgeCV, LassoCV, LassoLarsCV, ElasticNetCV):
            try:
                alphas = AlphaSelection(model())
            except YellowbrickTypeError:
                self.fail("could not instantiate RegressorCV on alpha selection")

    def test_only_regressors(self):
        """
        Assert AlphaSelection only works with regressors
        """
        with self.assertRaises(YellowbrickTypeError):
            model = AlphaSelection(SVC())

    def test_store_cv_values(self):
        """
        Assert that store_cv_values is true on RidgeCV
        """

        model = AlphaSelection(RidgeCV())
        self.assertTrue(model.estimator.store_cv_values)

        model = AlphaSelection(RidgeCV(store_cv_values=True))
        self.assertTrue(model.estimator.store_cv_values)

        model = AlphaSelection(RidgeCV(store_cv_values=False))
        self.assertTrue(model.estimator.store_cv_values)

    def test_get_alphas_param(self):
        """
        Assert that on known models we can get the alphas
        """
        alphas = np.logspace(-10, -2, 100)

        # Test original CV models
        for model in (RidgeCV, LassoCV, ElasticNetCV):
            try:
                model = AlphaSelection(model(alphas=alphas))
                malphas = model._find_alphas_param()
                self.assertTrue(np.array_equal(alphas, malphas))
            except YellowbrickValueError:
                self.fail("could not find alphas on {}".format(model.name))

        # Test LassoLars
        model = AlphaSelection(LassoLarsCV())
        model.fit(X, y)
        try:
            malphas = model._find_alphas_param()
            self.assertTrue(len(malphas) > 0)
        except YellowbrickValueError:
            self.fail("could not find alphas on {}".format(model.name))

    def test_get_errors_param(self):
        """
        Assert that on known models we can get the errors
        """

        # Test original CV models
        for model in (RidgeCV, LassoCV, LassoLarsCV, ElasticNetCV):
            try:
                model = AlphaSelection(model())
                model.fit(X, y)
                errors = model._find_errors_param()
                self.assertTrue(len(errors) > 0)
            except YellowbrickValueError:
                self.fail("could not find errors on {}".format(model.name))
