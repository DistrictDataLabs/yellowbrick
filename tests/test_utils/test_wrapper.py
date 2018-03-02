# tests.test_utils.test_wrapper
# Testing for the wrapping utility.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Mon May 22 09:25:52 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_wrapper.py [b2ecd50] benjamin@bengfort.com $

"""
Testing for the wrapping utility.
"""

##########################################################################
## Imports
##########################################################################

import unittest

from yellowbrick.base import Visualizer
from yellowbrick.utils.wrapper import *
from sklearn.naive_bayes import MultinomialNB

try:
    from unittest import mock
except ImportError:
    import mock


##########################################################################
## Fixture
##########################################################################

class MockVisualizer(Visualizer):

    def __init__(self, ax=None, **kwargs):
        self.ax = ax
        self.fit = mock.MagicMock()
        self.finalize = mock.MagicMock()
        self.poof = mock.MagicMock()
        self.set_title = mock.MagicMock()

    @property
    def ax(self):
        return self._ax

    @ax.setter
    def ax(self, val):
        self._ax = val


class WrappedEstimator(MockVisualizer, Wrapper):

    def __init__(self, **kwargs):
        self.estimator = mock.MagicMock(spec=MultinomialNB())

        Wrapper.__init__(self, self.estimator)
        MockVisualizer.__init__(self, **kwargs)

    def draw(self):
        return True

    def foo(self, a, b):
        return a+b


##########################################################################
## Wrapper Test Case
##########################################################################

class WrapperTests(unittest.TestCase):

    def test_wrapper_methods(self):
        """
        Asssert that local wrapper methods are called
        """
        obj = WrappedEstimator()

        # Assert that all the wrapper methods are called
        self.assertTrue(obj.draw())
        self.assertEqual(obj.foo(2,2), 4)
        self.assertIsNotNone(obj.estimator)

    def test_super_methods(self):
        """
        Assert that Visualizer super methods are called
        """

        obj = WrappedEstimator()

        # Assert that visualizer methods are called
        obj.fit()
        obj.finalize()
        obj.poof()
        obj.set_title()

        self.assertIsNone(obj.ax)
        obj.fit.assert_called_once_with()
        obj.finalize.assert_called_once_with()
        obj.poof.assert_called_once_with()
        obj.set_title.assert_called_once_with()

    def test_wrapped_methods(self):
        """
        Assert that wrapped estimator methods are calle d
        """
        obj = WrappedEstimator()

        # Assert that estimator methods are called
        obj.predict()
        obj.predict_proba()
        obj.score()

        obj._wrapped.predict.assert_called_once_with()
        obj._wrapped.predict_proba.assert_called_once_with()
        obj._wrapped.score.assert_called_once_with()
