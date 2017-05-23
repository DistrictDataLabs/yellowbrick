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

import unittest

from yellowbrick.utils.helpers import *

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import LSHForest
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import AffinityPropagation, Birch



##########################################################################
## Helper Function Tests
##########################################################################

class HelpersTests(unittest.TestCase):

    ##////////////////////////////////////////////////////////////////////
    ## get_model_name testing
    ##////////////////////////////////////////////////////////////////////

    def test_real_model(self):
        """
        Test that model name works for sklearn estimators
        """
        model1 = LassoCV()
        model2 = LSHForest()
        model3 = KMeans()
        model4 = RandomForestClassifier()
        self.assertEqual(get_model_name(model1), 'LassoCV')
        self.assertEqual(get_model_name(model2), 'LSHForest')
        self.assertEqual(get_model_name(model3), 'KMeans')
        self.assertEqual(get_model_name(model4), 'RandomForestClassifier')

    def test_pipeline(self):
        """
        Test that model name works for sklearn pipelines
        """
        pipeline = Pipeline([('reduce_dim', PCA()),
                             ('linreg', LinearRegression())])
        self.assertEqual(get_model_name(pipeline), 'LinearRegression')

    def test_int_input(self):
        """
        Assert a type error is raised when an int is passed to model name.
        """
        self.assertRaises(TypeError, get_model_name, 1)

    def test_str_input(self):
        """
        Assert a type error is raised when a str is passed to model name.
        """
        self.assertRaises(TypeError, get_model_name, 'helloworld')


##########################################################################
## Numeric Function Tests
##########################################################################


class DivSafeTests(unittest.TestCase):

    def test_div_1d_by_scalar(self):
        result = div_safe( [-1, 0, 1], 0 )
        self.assertTrue(result.all() == 0)

    def test_div_1d_by_1d(self):
        result =div_safe( [-1, 0 , 1], [0,0,0])
        self.assertTrue(result.all() == 0)

    def test_div_2d_by_1d(self):
        numerator = np.array([[-1,0,1,2],[1,-1,0,3]])
        denominator = [0,0,0,0]
        result = div_safe(numerator, denominator)

    def test_invalid_dimensions(self):
            numerator = np.array([[-1,0,1,2],[1,-1,0,3]])
            denominator = [0,0]
            with self.assertRaises(ValueError):
                result = div_safe(numerator, denominator)

    def test_div_scalar_by_scalar(self):
        with self.assertRaises(ValueError):
            result = div_safe(5, 0)


##########################################################################
## Features/Array Tests
##########################################################################

class NarrayIntColumnsTests(unittest.TestCase):

    def test_has_ndarray_int_columns_true_int_features(self):
        x = np.random.rand(3,5)
        features = [0, 1]
        self.assertTrue(has_ndarray_int_columns(features, x))

    def test_has_ndarray_int_columns_true_int_strings(self):
        x = np.random.rand(3,5)
        features = ['0', '1']
        self.assertTrue(has_ndarray_int_columns(features, x))

    def test_has_ndarray_int_columns_false_not_numeric(self):
        x = np.random.rand(3,5)
        features = ['a', '1']
        self.assertFalse(has_ndarray_int_columns(features, x))

    def test_has_ndarray_int_columns_false_outside_column_range(self):
        x = np.random.rand(3,5)
        features = ['0', '10']
        self.assertFalse(has_ndarray_int_columns(features, x))


##########################################################################
## String Helpers Tests
##########################################################################

class StringHelpersTests(unittest.TestCase):

    def test_slugifiy(self):
        """
        Test the slugify helper utility
        """

        cases = (
            ("This is a test ---", "this-is-a-test"),
            ("This -- is a ## test ---" , "this-is-a-test"),
        )

        for case, expected in cases:
            self.assertEqual(expected, slugify(case))


##########################################################################
## Execute Tests
##########################################################################

if __name__ == "__main__":
    unittest.main()
