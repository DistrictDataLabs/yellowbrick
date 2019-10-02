# tests.test_utils.test_wrapper
# Testing for the wrapping utility.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Mon May 22 09:25:52 2017 -0700
#
# Copyright (C) 2017 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_wrapper.py [b2ecd50] benjamin@bengfort.com $

"""
Testing for the wrapping utility.
"""

##########################################################################
## Imports
##########################################################################

from unittest import mock

from yellowbrick.base import Visualizer
from yellowbrick.utils.wrapper import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB


##########################################################################
## Fixture
##########################################################################


class MockVisualizer(Visualizer):
    def __init__(self, ax=None, **kwargs):
        self.ax = ax
        self.fit = mock.MagicMock()
        self.finalize = mock.MagicMock()
        self.show = mock.MagicMock()
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
        return a + b


##########################################################################
## Wrapper Test Case
##########################################################################


class TestWrapper(object):
    """
    Test the object Wrapper mixin utility
    """

    def test_wrapper_methods(self):
        """
        Asssert that local wrapper methods are called
        """
        obj = WrappedEstimator()

        # Assert that all the wrapper methods are called
        assert obj.draw()
        assert obj.foo(2, 2) == 4
        assert obj.estimator is not None

    def test_super_methods(self):
        """
        Assert that Visualizer super methods are called
        """

        obj = WrappedEstimator()

        # Assert that visualizer methods are called
        obj.fit()
        obj.finalize()
        obj.show()
        obj.set_title()

        assert obj.ax is None
        obj.fit.assert_called_once_with()
        obj.finalize.assert_called_once_with()
        obj.show.assert_called_once_with()
        obj.set_title.assert_called_once_with()

    def test_wrapped_methods(self):
        """
        Assert that wrapped estimator methods are called
        """
        obj = WrappedEstimator()

        # Assert that estimator methods are called
        obj.predict()
        obj.predict_proba()
        obj.score()

        obj._wrapped.predict.assert_called_once_with()
        obj._wrapped.predict_proba.assert_called_once_with()
        obj._wrapped.score.assert_called_once_with()

    def test_rewrap_object(self):
        """
        Test the ability to "rewrap" an object on demand
        """
        obj = WrappedEstimator()
        old = obj._wrapped
        new = mock.MagicMock(spec=GaussianNB())

        obj.predict()
        old.predict.assert_called_once()
        new.assert_not_called()

        # rewrap
        obj._wrapped = new
        obj.predict()
        old.predict.assert_called_once()
        new.predict.assert_called_once()
