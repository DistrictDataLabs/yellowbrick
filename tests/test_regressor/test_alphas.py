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

import sys
import pytest
import numpy as np

from tests.base import VisualTestCase
from numpy.testing.utils import assert_array_equal

from yellowbrick.regressor.alphas import *
from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.svm import SVR, SVC
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LassoLars, LassoLarsCV
from sklearn.linear_model import ElasticNet, ElasticNetCV


##########################################################################
## Alpha Selection Tests
##########################################################################

class TestAlphaSelection(VisualTestCase):
    """
    Test the AlphaSelection visualizer
    """

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_similar_image(self):
        """
        Integration test with image simiarlity comparison
        """

        visualizer = AlphaSelection(LassoCV(random_state=0))

        X, y = make_regression(random_state=0)
        visualizer.fit(X, y)
        visualizer.poof()

        self.assert_images_similar(visualizer)

    def test_regressor_cv(self):
        """
        Ensure only "CV" regressors are allowed
        """
        # TODO: parametrize with models when unittest dependency removed
        for model in (SVR, Ridge, Lasso, LassoLars, ElasticNet):
            with pytest.raises(YellowbrickTypeError):
                AlphaSelection(model())

        # TODO: parametrize with models when unittest dependency removed (new test case)
        for model in (RidgeCV, LassoCV, LassoLarsCV, ElasticNetCV):
            try:
                AlphaSelection(model())
            except YellowbrickTypeError:
                pytest.fail("could not instantiate RegressorCV on alpha selection")

    def test_only_regressors(self):
        """
        Assert AlphaSelection only works with regressors
        """
        # TODO: parameterize with classifier, clusterer, decomposition
        with pytest.raises(YellowbrickTypeError):
            AlphaSelection(SVC())

    def test_store_cv_values(self):
        """
        Assert that store_cv_values is true on RidgeCV
        """

        model = AlphaSelection(RidgeCV())
        assert model.estimator.store_cv_values

        model = AlphaSelection(RidgeCV(store_cv_values=True))
        assert model.estimator.store_cv_values

        model = AlphaSelection(RidgeCV(store_cv_values=False))
        assert model.estimator.store_cv_values

    def test_get_alphas_param(self):
        """
        Assert that we can get the alphas from ridge, lasso, and elasticnet
        """
        alphas = np.logspace(-10, -2, 100)

        # Test original CV models
        # TODO: parametrize this test with different models
        for model in (RidgeCV, LassoCV, ElasticNetCV):
            try:
                model = AlphaSelection(model(alphas=alphas))
                malphas = model._find_alphas_param()
                assert_array_equal(alphas, malphas)
            except YellowbrickValueError:
                pytest.fail("could not find alphas on {}".format(model.name))

    def test_get_alphas_param_lassolars(self):
        """
        Assert that we can get alphas from lasso lars.
        """
        X, y = make_regression()
        model = AlphaSelection(LassoLarsCV())
        model.fit(X, y)
        try:
            malphas = model._find_alphas_param()
            assert len(malphas) > 0
        except YellowbrickValueError:
            pytest.fail("could not find alphas on {}".format(model.name))

    def test_get_errors_param(self):
        """
        Test known models we can get the cv errors for alpha selection
        """

        # Test original CV models
        # TODO: parametrize this test with different models
        for model in (RidgeCV, LassoCV, LassoLarsCV, ElasticNetCV):
            try:
                model = AlphaSelection(model())

                X, y = make_regression()
                model.fit(X, y)

                errors = model._find_errors_param()
                assert len(errors) > 0
            except YellowbrickValueError:
                pytest.fail("could not find errors on {}".format(model.name))

    def test_score(self):
        """
        Assert the score method returns an R2 value
        """
        visualizer = AlphaSelection(RidgeCV())

        X, y = make_regression(random_state=352)
        visualizer.fit(X, y)
        assert visualizer.score(X, y) == pytest.approx(0.9999780266590336)
