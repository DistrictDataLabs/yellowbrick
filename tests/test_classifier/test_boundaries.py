# tests.test_boundaries
# Ensure that the decision boundary visualizations work.
#
# Author:   Author:   Nathan Danielsen <nathan.danielsen@gmail.com>
# Created:  Sun Mar 19 13:01:29 2017 -0400
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_boundaries.py [] nathan.danielsen@gmail.com $
"""
Ensure that the Decision Boundary visualizations work.
"""

##########################################################################
# Imports
##########################################################################

import pytest
import numpy as np

from tests.base import VisualTestCase
from yellowbrick.classifier import *
from yellowbrick.exceptions import YellowbrickTypeError
from yellowbrick.exceptions import YellowbrickValueError

from sklearn import datasets
from sklearn import neighbors
from sklearn import naive_bayes

try:
    from unittest import mock
except ImportError:
    import mock

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
# Data
##########################################################################

X = np.array([
    [2.318, 2.727, 4.260, 7.212, 4.792, ],
    [2.315, 2.726, 4.295, 7.140, 4.783, ],
    [2.315, 2.724, 4.260, 7.135, 4.779, ],
    [2.110, 3.609, 4.330, 7.985, 5.595, ],
    [2.110, 3.626, 4.330, 8.203, 5.621, ],
    [2.110, 3.620, 4.470, 8.210, 5.612, ],
    [2.318, 2.727, 4.260, 7.212, 4.792, ],
    [2.315, 2.726, 4.295, 7.140, 4.783, ],
    [2.315, 2.724, 4.260, 7.135, 4.779, ],
    [2.110, 3.609, 4.330, 7.985, 5.595, ],
    [2.110, 3.626, 4.330, 8.203, 5.621, ],
    [2.110, 3.620, 4.470, 8.210, 5.612, ]
    ])

y = np.array([1, 2, 1, 2, 1, 0, 0, 1, 3, 1, 3, 2])

X_two_cols = X[:, :2]

##########################################################################
# Residuals Plots test case
##########################################################################


class DecisionBoundariesVisualizerTest(VisualTestCase):
    """
    DecisionBoundariesVisualizer 
    """

    def test_decision_bounardies(self):
        """
        Assert no errors during kNN DecisionBoundariesVisualizer integration
        """
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionViz(model)
        viz.fit_draw_poof(X_two_cols, y=y)

    def test_init(self):
        """
        Test correct initialization of the internal state
        """
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(model)

        self.assertEquals(viz.step_size, 0.0025)
        self.assertEqual(viz.name, 'KNeighborsClassifier')
        self.assertEqual(viz.estimator, model)

        self.assertIsNone(viz.classes_)
        self.assertIsNone(viz.features_)
        self.assertIsNotNone(viz.markers)
        self.assertIsNotNone(viz.scatter_alpha)
        self.assertTrue(viz.show_scatter)

        self.assertIsNone(viz.Z)
        self.assertIsNone(viz.xx)
        self.assertIsNone(viz.yy)
        self.assertIsNone(viz.class_labels)
        self.assertIsNone(viz.title)
        self.assertIsNone(viz.x)
        self.assertIsNone(viz.y)


    def test_scatter_xy_and_features_raise_error(self):
        """
        Assert that x,y and features will raise error
        """
        model = neighbors.KNeighborsClassifier(3)
        features = ["temperature", "relative_humidity", "light"]

        with self.assertRaises(YellowbrickValueError):
            DecisionBoundariesVisualizer(
                model, features=features, x='one', y='two'
            )

    def test_scatter_xy_changes_to_features(self):
        """
        Assert that x,y and features will raise error
        """
        model = neighbors.KNeighborsClassifier(3)
        visualizer = DecisionBoundariesVisualizer(model, x='one', y='two')
        self.assertEquals(visualizer.features_, ['one', 'two'])


    def test_fit(self):
        """
        Testing the fit method works as expected
        """
        model = neighbors.KNeighborsClassifier(3)
        model.fit = mock.MagicMock()
        model.predict = mock.MagicMock()

        viz = DecisionBoundariesVisualizer(model)
        fitted_viz = viz.fit(X_two_cols, y=y)

        # assert that classes and labels are established
        self.assertEqual(fitted_viz.classes_, {0: '0', 1: '1', 2: '2', 3: '3'})
        self.assertEqual(fitted_viz.features_, ['Feature One', 'Feature Two'])

        # assert that the fit method is called
        model.fit.assert_called_once_with(X_two_cols, y)
        # mock object is called twice in predict and reshape
        self.assertEqual(len(model.predict.mock_calls), 2)

        # test that attrs are set
        self.assertIsNotNone(fitted_viz.ax)
        self.assertIsNotNone(fitted_viz.Z_shape)

    def test_fit_class_labels(self):
        """
        Test fit with class labels specified
        """
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(
            model, classes=['one', 'two', 'three', 'four'])
        fitted_viz = viz.fit(X_two_cols, y=y)
        self.assertEquals(fitted_viz.classes_,
                          {'three': '2',
                           'four': '3',
                           'two': '1',
                           'one': '0'})

    def test_fit_class_labels_class_names_edge_case(self):
        """
        Edge case that more class labels are defined than in datatset
        """
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(
            model, classes=['one', 'two', 'three', 'four', 'five'])
        self.assertRaises(YellowbrickTypeError, viz.fit, X_two_cols, y=y)

    def test_fit_features_assignment_None(self):
        """
        Test fit when features is None
        """
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(model)
        self.assertIsNone(viz.features_)
        fitted_viz = viz.fit(X_two_cols, y=y)
        self.assertEquals(fitted_viz.features_, ['Feature One', 'Feature Two'])

    def test_fit_features_assignment(self):
        """
        Test fit when features are specified
        """
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(model, features=['one', 'two'])
        fitted_viz = viz.fit(X_two_cols, y=y)
        self.assertEquals(fitted_viz.features_, ['one', 'two'])

    @mock.patch("yellowbrick.classifier.boundaries.OrderedDict")
    def test_draw_ordereddict_calls(self, mock_odict):
        """
        Test draw with calls to ordered dict
        """
        mock_odict.return_value = {}
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(model, features=['one', 'two'])
        self.assertRaises(KeyError, viz.fit_draw, X_two_cols, y=y)
        self.assertEquals(len(mock_odict.mock_calls), 2)

    @mock.patch("yellowbrick.classifier.boundaries.resolve_colors")
    def test_draw_ordereddict_calls_one(self, mock_resolve_colors):
        """
        Test ordered dict calls resolve colors
        """
        mock_resolve_colors.return_value = []
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(model, features=['one', 'two'])
        self.assertRaises(StopIteration, viz.fit_draw, X_two_cols, y=y)
        self.assertEquals(len(mock_resolve_colors.mock_calls), 1)

    def test_draw_ax_show_scatter_true(self):
        """
        Test that the matplotlib functions are being called
        """
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(model, features=['one', 'two'])
        fitted_viz = viz.fit(X_two_cols, y=y)
        fitted_viz.ax = mock.Mock()
        fitted_viz.ax.pcolormesh = mock.MagicMock()
        fitted_viz.ax.scatter = mock.MagicMock()
        fitted_viz.ax.legend = mock.MagicMock()

        fitted_viz.draw(X_two_cols, y=y)
        self.assertEquals(len(fitted_viz.ax.pcolormesh.mock_calls), 1)
        self.assertEquals(len(fitted_viz.ax.scatter.mock_calls), 4)
        self.assertEquals(len(fitted_viz.ax.legend.mock_calls), 0)

    def test_draw_ax_show_scatter_False(self):
        """
        Test that the matplotlib called when the scatter plot isn't drawn
        """
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(
            model, features=['one', 'two'], show_scatter=False)
        fitted_viz = viz.fit(X_two_cols, y=y)
        fitted_viz.ax = mock.Mock()
        fitted_viz.ax.pcolormesh = mock.MagicMock()
        fitted_viz.ax.scatter = mock.MagicMock()
        fitted_viz.ax.legend = mock.MagicMock()

        fitted_viz.draw(X_two_cols, y=y)
        self.assertEquals(len(fitted_viz.ax.pcolormesh.mock_calls), 1)
        self.assertEquals(len(fitted_viz.ax.scatter.mock_calls), 0)
        self.assertEquals(len(fitted_viz.ax.legend.mock_calls), 1)

    def test_finalize(self):
        """
        Test the finalize method works as expected
        """
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(
            model, features=['one', 'two'], show_scatter=False)
        fitted_viz = viz.fit(X_two_cols, y=y)
        fitted_viz.draw(X_two_cols, y=y)

        fitted_viz.ax = mock.Mock()
        fitted_viz.ax.legend = mock.MagicMock()
        fitted_viz.ax.set_xlabel = mock.MagicMock()
        fitted_viz.ax.set_ylabel = mock.MagicMock()

        fitted_viz.poof()

        fitted_viz.ax.legend.assert_called_once_with(loc='best', frameon=True)
        fitted_viz.ax.set_xlabel.assert_called_once_with('one')
        fitted_viz.ax.set_ylabel.assert_called_once_with('two')

    def test_fit_draw(self):
        """
        Test fit draw shortcut
        """
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(
            model, features=['one', 'two'], show_scatter=False)

        viz.fit = mock.Mock()
        viz.draw = mock.Mock()

        viz.fit_draw(X_two_cols, y=y)

        viz.fit.assert_called_once_with(X_two_cols, y)
        viz.draw.assert_called_once_with(X_two_cols, y)

    def test_fit_draw_poof(self):
        """
        Test fit draw poof shortcut
        """
        model = neighbors.KNeighborsClassifier(3)
        viz = DecisionBoundariesVisualizer(
            model, features=['one', 'two'], show_scatter=False)

        viz.fit = mock.Mock()
        viz.draw = mock.Mock()
        viz.poof = mock.Mock()

        viz.fit_draw_poof(X_two_cols, y=y)

        viz.fit.assert_called_once_with(X_two_cols, y)
        viz.draw.assert_called_once_with(X_two_cols, y)
        viz.poof.assert_called_once_with()


    def test_integrated_plot_numpy_named_arrays(self):
        """
        Test integration of visualizer with numpy named arrays
        """
        model = naive_bayes.MultinomialNB()

        X = np.array([
             (1.1, 9.52, 1.23, 0.86, 7.89, 0.13),
             (3.4, 2.84, 8.65, 0.45, 7.43, 0.16),
             (1.2, 3.22, 6.56, 0.24, 3.45, 0.17),
             (3.8, 6.18, 2.45, 0.28, 2.53, 0.13),
             (5.1, 9.12, 1.06, 0.19, 1.43, 0.13),
             (4.4, 8.84, 4.97, 0.98, 1.35, 0.13),
             (3.2, 3.22, 5.03, 0.68, 3.53, 0.32),
             (7.8, 2.18, 6.87, 0.35, 3.25, 0.38),
            ], dtype=[('a','<f8'), ('b','<f8'),
                ('c','<f8'), ('d','<f8'),
                ('e','<f8'), ('f','<f8')]
        )

        y = np.array([1, 1, 0, 1, 0, 0, 1, 0])

        visualizer = DecisionBoundariesVisualizer(model, features=['a', 'f'])
        visualizer.fit_draw_poof(X, y=y)
        self.assertEquals(visualizer.features_, ['a', 'f'])
        self.assert_images_similar(visualizer)

    def test_integrated_scatter_numpy_arrays_no_names(self):
        """
        Test integration of visualizer with numpy arrays
        """
        model = neighbors.KNeighborsClassifier(3)

        visualizer = DecisionBoundariesVisualizer(model, features=[1, 2])
        visualizer.fit_draw_poof(X, y)
        self.assertEquals(visualizer.features_, [1, 2])

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_real_data_set_viz(self):
        """
        Test integration of visualizer with pandas dataset
        """
        model = naive_bayes.MultinomialNB()

        data = datasets.load_iris()
        feature_names = [name.replace(' ', '_') for name in  data.feature_names ]
        df = pd.DataFrame(data.data, columns=feature_names)
        X = df[['sepal_length_(cm)', 'sepal_width_(cm)']].as_matrix()
        y = data.target

        visualizer = DecisionBoundariesVisualizer(model)
        visualizer.fit_draw_poof(X, y)
        self.assert_images_similar(visualizer)

    @pytest.mark.skipif(pd is None, reason="test requires pandas")
    def test_quick_method(self):
        """
        Test quick function shortcut of visualizer
        """
        model = naive_bayes.MultinomialNB()

        data = datasets.load_iris()
        feature_names = [name.replace(' ', '_') for name in  data.feature_names ]
        df = pd.DataFrame(data.data, columns=feature_names)
        X = df[['sepal_length_(cm)', 'sepal_width_(cm)']].as_matrix()
        y = data.target

        decisionviz(model, X, y)
