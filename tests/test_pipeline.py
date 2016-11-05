# tests.test_pipeline
# Tests to ensure that the visual pipeline works as expected.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Oct 07 22:10:50 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_pipeline.py [] benjamin@bengfort.com $

"""
Tests to ensure that the visual pipeline works as expected.
"""

##########################################################################
## Imports
##########################################################################

import unittest

from yellowbrick.base import Visualizer
from yellowbrick.pipeline import VisualPipeline
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from unittest import mock
except ImportError:
    import mock


##########################################################################
## Mock Objects
##########################################################################

class Thing(object):
    pass


class EstimatorSpec(BaseEstimator):

    def fit(self, X, y=None, **kwargs):
        return self

MockEstimator = mock.Mock(spec = EstimatorSpec)


class TransformerSpec(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, **kwargs):
        return X

MockTransformer = mock.Mock(spec = TransformerSpec)


class VisualTransformerSpec(Visualizer, TransformerMixin):

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, **kwargs):
        return X

    def draw(self, **kwargs):
        pass

    def poof(self, **kwargs):
        pass

MockVisualTransformer = mock.Mock(spec = VisualTransformerSpec)


##########################################################################
## VisualPipeline Tests
##########################################################################

class VisualPipelineTests(unittest.TestCase):

    def test_validate_steps(self):
        """
        Assert that visual transformers can be added to pipelines
        """

        # Pipeline objects have a _validate_steps method that raises an
        # TypeError if the steps don't match transforms --> estimator.

        # validate a bad intermediate transformer on the Pipeline
        with self.assertRaises(TypeError):
            pipeline = Pipeline([
                ('real', MockTransformer()),
                ('bad', Thing()),
                ('model', MockEstimator()),
            ])

        # validate a bad intermediate transformer on the VisualPipeline
        with self.assertRaises(TypeError):
            pipeline = VisualPipeline([
                ('real', MockTransformer()),
                ('bad', Thing()),
                ('model', MockEstimator()),
            ])

        # validate a bad final estimator on the Pipeline
        with self.assertRaises(TypeError):
            pipeline = Pipeline([
                ('real', MockTransformer()),
                ('bad', Thing()),
            ])

        # validate a bad final estimator on the VisualPipeline
        with self.assertRaises(TypeError):
            pipeline = VisualPipeline([
                ('real', MockTransformer()),
                ('bad', Thing()),
            ])

        # validate visual transformers on a Pipeline
        try:
            pipeline = Pipeline([
                ('real', MockTransformer()),
                ('visual', MockVisualTransformer()),
                ('model', MockEstimator()),
            ])
        except TypeError:
            self.fail("could not add a visual transformer to a Pipeline!")

        # validate visual transformers on a VisualPipeline
        try:
            pipeline = VisualPipeline([
                ('real', MockTransformer()),
                ('visual', MockVisualTransformer()),
                ('model', MockEstimator()),
            ])
        except TypeError:
            self.fail("could not add a visual transformer to a VisualPipeline!")

    def test_visual_steps_property(self):
        """
        Test the visual steps property to filter visualizers
        """

        pipeline = VisualPipeline([
            ('a', MockTransformer()),
            ('b', VisualTransformerSpec()),
            ('c', MockTransformer()),
            ('d', VisualTransformerSpec()),
            ('e', MockEstimator()),
        ])

        self.assertIn('b', pipeline.visual_steps)
        self.assertIn('d', pipeline.visual_steps)

    @unittest.skip("mocks don't work for some reason")
    def test_fit_transform_poof_and_draw_calls(self):
        """
        Test calling fit, transform, draw and poof on the pipeline
        """

        pipeline = VisualPipeline([
            ('a', MockTransformer()),
            ('b', MockVisualTransformer()),
            ('c', MockTransformer()),
            ('d', MockVisualTransformer()),
            ('e', MockEstimator()),
        ])

        X = [[1, 1, 1, 1, 1],
             [2, 2, 2, 2, 2],
             [3, 3, 3, 3, 3]]

        y =  [1, 2, 3, 4, 5]

        pipeline.fit(X, y)
        for name, step in pipeline.named_steps.items():
            step.fit.assert_called_once_with(X, y)

        pipeline.transform(X)
        for name, step in pipeline.named_steps.items():
            step.transform.assert_called_once_with(X)

        pipeline.draw()
        for name, step in pipeline.named_steps.items():
            if name in {'a', 'c', 'e'}: continue
            step.draw.assert_called_once_with()

        pipeline.poof()
        for name, step in pipeline.named_steps.items():
            if name in {'a', 'c', 'e'}: continue
            step.poof.assert_called_once_with()
