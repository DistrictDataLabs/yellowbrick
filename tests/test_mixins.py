# tests.test_mixins.py
# Assertions for the high level mixins.
#
# Author:   Nathan Danielsen <rbilbro@gmail.com.com>
# Created:  Sat Mar 12 14:17:29 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt

"""
Assertions for the high level mixins.
"""

##########################################################################
## Imports
##########################################################################

import unittest

import numpy as np
from yellowbrick.mixins import *

try:
    from unittest import mock
except ImportError:
    import mock

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
## Mixin Tests
##########################################################################

class MixinTests(unittest.TestCase):
    """
    Test the high level mixins for yellowbrick
    """

    def test_mixin_no_errors_two_features(self):
        """
        Assert that all visualizers return self
        """
        class BaseTest(BivariateFeatureMixin, object): pass

        twofeaturemixin = BaseTest(features=[1,2])
        self.assertIn('BaseTest', str(twofeaturemixin))

    def test_mixin_raise_error_three_features(self):
        """
        Assert that all visualizers return self
        """
        class BaseTest(BivariateFeatureMixin, object): pass
        with self.assertRaises(YellowbrickValueError) as context:
            twofeaturemixin = BaseTest(features=['feature_one', 'feature_two', 'feature_three'])

    def test_mixin_raise_error_one_features(self):
        """
        Assert that all visualizers return self
        """
        class BaseTest(BivariateFeatureMixin, object): pass

        with self.assertRaises(YellowbrickValueError) as context:
            twofeaturemixin = BaseTest(features=['one_feature'])

    def test_mixin_no_features_pass_no_errors(self):
        """
        Assert that all visualizers return self
        """
        class BaseTest(BivariateFeatureMixin, object): pass
        twofeaturemixin = BaseTest()
        self.assertIn('BaseTest', str(twofeaturemixin))

    def test_mixin_inits(self):
        """
        Assert that all visualizers return self
        """
        twofeaturemixin = BivariateFeatureMixin()
        self.assertIn('BivariateFeatureMixin', str(twofeaturemixin))


    def test_mixin_fit(self):
        """Test that matrixes with only two colums are excepted """
        X_two_cols =  X[:,:2]
        class BaseTest(BivariateFeatureMixin, object): pass
        twofeaturemixin = BaseTest()
        twofeaturemixin.fit(X_two_cols)

    def test_mixin_fit_three_columns(self):
        """Test that matrixes with only two colums are excepted """
        X_three_cols =  X[:,:3]
        class BaseTest(BivariateFeatureMixin, object): pass
        twofeaturemixin = BaseTest()

        with self.assertRaises(YellowbrickValueError) as context:
            twofeaturemixin.fit(X_three_cols)
