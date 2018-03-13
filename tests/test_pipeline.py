# tests.test_pipeline
# Tests to ensure that the visual pipeline works as expected.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Oct 07 22:10:50 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_pipeline.py [1efae1f] benjamin@bengfort.com $

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


class MockEstimator(BaseEstimator):

    def fit(self, X, y=None, **kwargs):
        return self

class MockVisualEstimator(Visualizer):

    def fit(self, X, y=None, **kwargs):
        self.draw(**kwargs)
        return self

    def draw(self, **kwargs):
        pass


class MockTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, **kwargs):
        return X


class MockVisualTransformer(Visualizer, TransformerMixin):

    def fit(self, X, y=None, **kwargs):
        self.draw(**kwargs)
        return self

    def transform(self, X, **kwargs):
        return X

    def draw(self, **kwargs):
        pass


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
            Pipeline([
                ('real', MockTransformer()),
                ('bad', Thing()),
                ('model', MockEstimator()),
            ])

        # validate a bad intermediate transformer on the VisualPipeline
        with self.assertRaises(TypeError):
            VisualPipeline([
                ('real', MockTransformer()),
                ('bad', Thing()),
                ('model', MockEstimator()),
            ])

        # validate a bad final estimator on the Pipeline
        with self.assertRaises(TypeError):
            Pipeline([
                ('real', MockTransformer()),
                ('bad', Thing()),
            ])

        # validate a bad final estimator on the VisualPipeline
        with self.assertRaises(TypeError):
            VisualPipeline([
                ('real', MockTransformer()),
                ('bad', Thing()),
            ])

        # validate visual transformers on a Pipeline
        try:
            Pipeline([
                ('real', MockTransformer()),
                ('visual', MockVisualTransformer()),
                ('model', MockEstimator()),
            ])
        except TypeError:
            self.fail("could not add a visual transformer to a Pipeline!")

        # validate visual transformers on a VisualPipeline
        try:
            VisualPipeline([
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
            ('b', MockVisualTransformer()),
            ('c', MockTransformer()),
            ('d', MockVisualTransformer()),
            ('e', MockEstimator()),
        ])

        self.assertNotIn('a', pipeline.visual_steps)
        self.assertIn('b', pipeline.visual_steps)
        self.assertNotIn('c', pipeline.visual_steps)
        self.assertIn('d', pipeline.visual_steps)
        self.assertNotIn('e', pipeline.visual_steps)

    def test_pipeline_poof(self):
        """
        Test the poof call against the VisualPipeline
        """

        pipeline = VisualPipeline([
            ('a', mock.MagicMock(MockTransformer())),
            ('b', mock.MagicMock(MockVisualTransformer())),
            ('c', mock.MagicMock(MockTransformer())),
            ('d', mock.MagicMock(MockVisualTransformer())),
            ('e', mock.MagicMock(MockEstimator()),)
        ])

        pipeline.poof()
        pipeline.steps[1][1].poof.assert_called_once_with(outpath=None)
        pipeline.steps[3][1].poof.assert_called_once_with(outpath=None)

    def test_pipeline_savefig_poof(self):
        """
        Test the poof call with an outdir to save all the figures
        """
        pipeline = VisualPipeline([
            ('a', mock.MagicMock(MockTransformer())),
            ('b', mock.MagicMock(MockVisualTransformer())),
            ('c', mock.MagicMock(MockTransformer())),
            ('d', mock.MagicMock(MockVisualTransformer())),
            ('e', mock.MagicMock(MockVisualEstimator()),)
        ])

        pipeline.poof(outdir="/tmp/figures")
        pipeline.steps[1][1].poof.assert_called_once_with(outpath="/tmp/figures/b.pdf")
        pipeline.steps[3][1].poof.assert_called_once_with(outpath="/tmp/figures/d.pdf")
        pipeline.steps[4][1].poof.assert_called_once_with(outpath="/tmp/figures/e.pdf")

    @unittest.skip("need to find a way for fit to return self in mocks")
    def test_fit_transform_poof_and_draw_calls(self):
        """
        Test calling fit, transform, and poof on the pipeline
        """

        pipeline = VisualPipeline([
            ('a', mock.MagicMock(MockTransformer())),
            ('b', mock.MagicMock(MockVisualTransformer())),
            ('c', mock.MagicMock(MockTransformer())),
            ('d', mock.MagicMock(MockVisualTransformer())),
            ('e', mock.MagicMock(MockEstimator()),)
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
            if name == 'e': continue
            step.transform.assert_called_once_with(X)

        pipeline.poof()
        for name, step in pipeline.named_steps.items():
            if name in {'a', 'c', 'e'}: continue
            step.poof.assert_called_once_with(outpath=None)
