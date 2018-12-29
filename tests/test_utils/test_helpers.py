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

    @pytest.mark.parametrize("model, name", [
        (LassoCV, 'LassoCV'),
        (KNeighborsClassifier, 'KNeighborsClassifier'),
        (KMeans, 'KMeans'),
        (RandomForestClassifier, 'RandomForestClassifier'),
    ], ids=["LassoCV", "KNeighborsClassifier", "KMeans", "RandomForestClassifier"])
    def test_real_model(self, model, name):
        """
        Test getting model name for sklearn estimators
        """
        assert get_model_name(model()) == name

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

    def test_prop_to_size_list(self):
        """
        Test prop to size correctly returns scaled values for a list
        """
        # Hieghts (in cm) of U.S. Presidents in order of term until Lincoln
        heights = [188, 170, 189, 163, 183, 171, 185, 168, 173, 183, 173, 173, 175, 178, 183, 193]
        sizes = prop_to_size(heights, mi=1, ma=10, log=False, power=0.33)

        npt.assert_array_almost_equal(sizes, np.array([
            9.47447296,  6.56768746,  9.58486955,  1.        ,  8.87285756,
            6.81851544,  9.12441277,  5.98256068,  7.26314542,  8.87285756,
            7.26314542,  7.26314542,  7.65154152,  8.15982835,  8.87285756,
            10.
        ]))

    def test_prop_to_size_log(self):
        """
        Test prop to size returns natural log scaled values
        """
        # Hieghts (in cm) of U.S. Presidents in order of term until Lincoln
        heights = [188, 170, 189, 163, 183, 171, 185, 168, 173, 183, 173, 173, 175, 178, 183, 193]
        sizes = prop_to_size(heights, mi=1, ma=10, log=True, power=0.5)

        npt.assert_array_almost_equal(sizes, np.array([
            9.271337,  5.49004 ,  9.423692,  1.      ,  8.449214,  5.792968,
            8.791172,  4.806088,  6.343007,  8.449214,  6.343007,  6.343007,
            6.835994,  7.496806,  8.449214, 10.
        ]))

    def test_prop_to_size_default(self):
        """
        Test the default values of prop to size are correct
        """
        vals = np.random.normal(50, 23, 500)
        sizes = prop_to_size(vals)

        assert sizes.ndim == vals.ndim
        assert sizes.shape == vals.shape
        assert sizes.max() <= 5.0
        assert sizes.min() >= 0.0

    def test_prop_to_size_zero_division(self):
        """
        Ensure that prop to size does not cause division by zero errors
        """
        vals = [8]*8
        sizes = prop_to_size(vals)
        npt.assert_array_equal(sizes, [0]*8)


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

    @pytest.mark.parametrize("a, increasing", [
        (np.array([0.8]), True),
        (np.array([9]), False),
        (np.array([0.2, 1.3, 1.4, 1.4, 1.4, 1.5, 8.3, 8.5]), True),
        (np.array([8, 7, 6, 5, 5, 5, 5, 4, 3, -1, -5]), False),
    ], ids=["increasing single", "decreasing single", "increasing", "decreasing"])
    def test_is_monotonic(self, a, increasing):
        """
        Test if a vector is monotonic
        """
        assert is_monotonic(a, increasing)

    @pytest.mark.parametrize("a, increasing", [
        (np.array([0.2, 1.3, 1.3, 0.2, 1.8]), True),
        (np.array([8, 7, 7, 8, 9, 6, 5]), False),
    ], ids=["increasing", "decreasing"])
    def test_not_is_monotonic(self, a, increasing):
        """
        Test if a vector is not monotonic
        """
        assert not is_monotonic(a, increasing)

    def test_multi_dim_is_monotonic(self):
        """
        Assert monotonicity is not decidable on multi-dimensional array
        """
        with pytest.raises(ValueError):
            is_monotonic(np.array([[1,2,3], [4,5,6], [7,8,9]]))


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
