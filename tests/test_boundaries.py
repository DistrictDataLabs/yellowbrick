# tests.test_boundaries
# Ensure that the decision boundary visualizations work.
#
# Author:   Author:   Nathan Danielsen <nathan.danielsen@gmail.com>
# Created:  Sun Mar 19 13:01:29 2017 -0400
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_knn.py [] nathan.danielsen@gmail.com $

"""
Ensure that the KNN boundary visualizations work.
"""

##########################################################################
## Imports
##########################################################################

from unittest import mock
import unittest
import numpy as np

from tests.base import VisualTestCase
from yellowbrick.boundaries import *

from sklearn import neighbors


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

y = np.array([1, 2, 1, 2, 1, 0, 0, 1, 3, 1, 3, 2])

X_two_cols =  X[:,:2]
##########################################################################
## Residuals Plots test case
##########################################################################

class DecisionBoundariesVisualizerTest(VisualTestCase):

    def test_decision_bounardies(self):
        """
        Assert no errors occur during KnnDecisionBoundariesVisualizer integration
        """
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(model)
        viz.fit_draw_poof(X_two_cols, y=y)


    def test_init(self):
        """
        Testing the init method
        """
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(model)

        self.assertIn("#", viz.colors[0]) # default color palette
        self.assertEquals(viz.step_size, 0.0025)
        self.assertEqual(viz.name, 'KNeighborsClassifier')
        self.assertEqual(viz.estimator, model)

        self.assertIsNone(viz.classes_)
        self.assertIsNone(viz.features_)
        self.assertTrue(viz.show_scatter)

        self.assertIsNone(viz.Z)
        self.assertIsNone(viz.xx)
        self.assertIsNone(viz.yy)
        self.assertIsNone(viz.ax)
        self.assertIsNone(viz.class_labels)

    def test_fit(self):
        """
        Testing the fit method
        """
        model = neighbors.KNeighborsClassifier(3)
        model.fit = mock.MagicMock()
        model.predict = mock.MagicMock()

        viz = DecisionBoundariesVisualizer(model)
        fitted_viz =  viz.fit(X_two_cols, y=y)

        # assert that classes and labels are established
        self.assertEqual(fitted_viz.classes_, {0: '0', 1: '1', 2: '2', 3: '3'})
        self.assertEqual(fitted_viz.features_, ['0', '1'])

        # assert that the fit method is called
        model.fit.assert_called_once_with(X_two_cols, y)
        # mock object is called twice in predict and reshape
        self.assertEqual(len(model.predict.mock_calls), 2)

        # test that attrs are set
        self.assertIsNotNone(fitted_viz.ax)
        self.assertIsNotNone(fitted_viz.Z_shape)


    def test_fit_class_labels(self):
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(model, classes=['one', 'two', 'three', 'four'])
        fitted_viz = viz.fit(X_two_cols, y=y)
        self.assertEquals(fitted_viz.classes_, {'three': '2', 'four': '3', 'two': '1', 'one': '0'})

    def test_fit_class_labels_class_names_edge_case(self):
        """ Edge case that more class labels are defined than in datatset"""
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(model, classes=['one', 'two', 'three', 'four', 'five'])
        self.assertRaises(YellowbrickTypeError, viz.fit, X_two_cols, y=y )

    def test_fit_features_assignment_None(self):
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(model)
        self.assertIsNone(viz.features_)
        fitted_viz = viz.fit(X_two_cols, y=y)
        self.assertEquals(fitted_viz.features_, ['0', '1'])

    def test_fit_features_assignment(self):
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(model, features=['one', 'two'])
        fitted_viz = viz.fit(X_two_cols, y=y)
        self.assertEquals(fitted_viz.features_, ['one', 'two'])
