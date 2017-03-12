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

from yellowbrick.mixins import *

try:
    from unittest import mock
except ImportError:
    import mock

##########################################################################
## Imports
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
