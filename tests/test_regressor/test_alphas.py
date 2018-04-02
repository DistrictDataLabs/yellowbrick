# tests.test_regressor.test_alphas
# Tests for the alpha selection visualizations.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Tue Mar 07 12:13:04 2017 -0500
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_alphas.py [7d3f5e6] benjamin@bengfort.com $

"""
Tests for the alpha selection visualizations.
"""

##########################################################################
## Imports
##########################################################################

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
from sklearn.datasets import make_regression


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
                AlphaSelection(model())

        for model in (RidgeCV, LassoCV, LassoLarsCV, ElasticNetCV):
            try:
                AlphaSelection(model())
            except YellowbrickTypeError:
                self.fail("could not instantiate RegressorCV on alpha selection")

    def test_only_regressors(self):
        """
        Assert AlphaSelection only works with regressors
        """
        with self.assertRaises(YellowbrickTypeError):
            AlphaSelection(SVC())

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
        Assert that we can get the alphas from ridge, lasso, and elasticnet
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

    def test_get_alphas_param_lassolars(self):
        """
        Assert that we can get alphas from lasso lars.
        """
        X, y = make_regression()
        model = AlphaSelection(LassoLarsCV())
        model.fit(X, y)
        try:
            malphas = model._find_alphas_param()
            self.assertTrue(len(malphas) > 0)
        except YellowbrickValueError:
            self.fail("could not find alphas on {}".format(model.name))

    def test_get_errors_param(self):
        """
        Test known models we can get the cv errors for alpha selection
        """

        # Test original CV models
        for model in (RidgeCV, LassoCV, LassoLarsCV, ElasticNetCV):
            try:
                model = AlphaSelection(model())

                X, y = make_regression()
                model.fit(X, y)

                errors = model._find_errors_param()
                self.assertTrue(len(errors) > 0)
            except YellowbrickValueError:
                self.fail("could not find errors on {}".format(model.name))


    def test_similar_image(self):
        """
        Test similar plot drawn
        """

        visualizer = AlphaSelection(LassoCV(random_state=0))

        X, y = make_regression(random_state=0)
        visualizer.fit(X, y)
        visualizer.poof()

        self.assert_images_similar(visualizer)
