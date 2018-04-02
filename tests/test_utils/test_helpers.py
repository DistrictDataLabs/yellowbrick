# -*- coding: utf-8 -*-
# tests.test_utils.test_helpers
# Tests for the stand alone helper functions in Yellowbrick utils.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri May 19 10:43:43 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_helpers.py [79cd8cf] benjamin@bengfort.com $

"""
Tests for the stand alone helper functions in Yellowbrick utils.
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np
import numpy.testing as npt

from yellowbrick.utils.helpers import *

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


##########################################################################
## Helper Function Tests
##########################################################################

class TestHelpers(object):
    """
    Helper functions and utilities
    """

    def test_real_model(self):
        """
        Test getting model name for sklearn estimators
        """
        model1 = LassoCV()
        model2 = KNeighborsClassifier()
        model3 = KMeans()
        model4 = RandomForestClassifier()
        assert get_model_name(model1) == 'LassoCV'
        assert get_model_name(model2) == 'KNeighborsClassifier'
        assert get_model_name(model3) == 'KMeans'
        assert get_model_name(model4) == 'RandomForestClassifier'

    def test_pipeline(self):
        """
        Test getting model name for sklearn pipelines
        """
        pipeline = Pipeline([('reduce_dim', PCA()),
                             ('linreg', LinearRegression())])
        assert get_model_name(pipeline) == 'LinearRegression'

    def test_int_input(self):
        """
        Assert a type error is raised when an int is passed to model name
        """
        with pytest.raises(TypeError):
            get_model_name(1)

    def test_str_input(self):
        """
        Assert a type error is raised when a str is passed to model name
        """
        with pytest.raises(TypeError):
            get_model_name('helloworld')


##########################################################################
## Numeric Function Tests
##########################################################################


class TestNumericFunctions(object):
    """
    Numeric helper functions
    """

    def test_div_1d_by_scalar(self):
        """
        Test divide 1D vector by scalar
        """
        result = div_safe( [-1, 0, 1], 0 )
        assert result.all() == 0

    def test_div_1d_by_1d(self):
        """
        Test divide 1D vector by another 1D vector with same length
        """
        result = div_safe( [-1, 0 , 1], [0,0,0])
        assert result.all() == 0

    def test_div_2d_by_1d(self):
        """
        Test divide 2D vector by 1D vector with similar shape component
        """
        numerator = np.array([[-1,0,1,2],[1,-1,0,3]])
        denominator = [0,0,0,0]
        npt.assert_array_equal(
            div_safe(numerator, denominator),
            np.array([[0,0,0,0], [0,0,0,0]])
        )

    def test_invalid_dimensions(self):
        """
        Assert an error is raised on division with invalid dimensions
        """
        numerator = np.array([[-1,0,1,2],[1,-1,0,3]])
        denominator = [0,0]
        with pytest.raises(ValueError):
            div_safe(numerator, denominator)

    def test_div_scalar_by_scalar(self):
        """
        Assert a value error is raised when trying to divide two scalars
        """
        with pytest.raises(ValueError):
            div_safe(5, 0)


##########################################################################
## Features/Array Tests
##########################################################################

class TestNarrayIntColumns(object):
    """
    Features and array helper tests
    """

    def test_has_ndarray_int_columns_true_int_features(self):
        """
        Ensure ndarray with int features has int columns
        """
        x = np.random.rand(3,5)
        features = [0, 1]
        assert has_ndarray_int_columns(features, x)

    def test_has_ndarray_int_columns_true_int_strings(self):
        """
        Ensure ndarray with str(int) features has int columns
        """
        x = np.random.rand(3,5)
        features = ['0', '1']
        assert has_ndarray_int_columns(features, x)

    def test_has_ndarray_int_columns_false_not_numeric(self):
        """
        Ensure ndarray with str features does not have int columns
        """
        x = np.random.rand(3,5)
        features = ['a', '1']
        assert not has_ndarray_int_columns(features, x)

    def test_has_ndarray_int_columns_false_outside_column_range(self):
        """
        Ensure ndarray with str(int) outside range does not have int columns
        """
        x = np.random.rand(3,5)
        features = ['0', '10']
        assert not has_ndarray_int_columns(features, x)


##########################################################################
## String Helpers Tests
##########################################################################

class TestStringHelpers(object):
    """
    String helper functions
    """

    def test_slugifiy(self):
        """
        Test the slugify helper utility
        """

        cases = (
            ("This is a test ---", "this-is-a-test"),
            ("This -- is a ## test ---" , "this-is-a-test"),
        )

        for case, expected in cases:
            assert expected == slugify(case)
