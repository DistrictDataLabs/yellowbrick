# tests.test_utils
# Test the yellowbrick utilities module.
#
# Author:   Jason Keung <jason.s.keung@gmail.com>
# Author:   Patrick O'Melveny <pvomelveny@gmail.com>
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Jun 02 15:33:18 2016 -0500
#
# Copyright (C) 2016 District Data LAbs
# For license information, see LICENSE.txt
#
# ID: test_utils.py [] jason.s.keung@gmail.com $

"""
Test the yellowbrick utilities module.
"""

##########################################################################
## Imports
##########################################################################

import inspect
import unittest

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import LSHForest
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from yellowbrick.utils import *
from yellowbrick.base import Visualizer, ScoreVisualizer, ModelVisualizer


##########################################################################
## Model Utility Tests
##########################################################################

class ModelUtilityTests(unittest.TestCase):

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
        self.assertEqual(get_model_name(model1), 'LassoCV')
        self.assertEqual(get_model_name(model2), 'LSHForest')
        self.assertEqual(get_model_name(model3), 'KMeans')

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

    ##////////////////////////////////////////////////////////////////////
    ## isestimator testing
    ##////////////////////////////////////////////////////////////////////

    def test_estimator_alias(self):
        """
        Assert is_estimator aliases isestimator
        """
        self.assertEqual(
            is_estimator(LinearRegression), isestimator(LinearRegression)
        )

    def test_estimator_instance(self):
        """
        Test that isestimator works for instances
        """

        models = (
            LinearRegression(),
            LogisticRegression(),
            KMeans(),
            LSHForest(),
            PCA(),
            LassoCV(),
            RandomForestClassifier(),
        )

        for model in models:
            self.assertTrue(isestimator(model))

    def test_pipeline_instance(self):
        """
        Test that isestimator works for pipelines
        """
        model = Pipeline([
            ('reduce_dim', PCA()),
            ('linreg', LinearRegression())
        ])

        self.assertTrue(isestimator(model))

    def test_estimator_class(self):
        """
        Test that isestimator works for classes
        """
        models = (
            LinearRegression,
            LogisticRegression,
            KMeans,
            LSHForest,
            PCA,
            LassoCV,
            RandomForestClassifier,
        )

        for model in models:
            self.assertTrue(inspect.isclass(model))
            self.assertTrue(isestimator(model))

    def test_collection_not_estimator(self):
        """
        Make sure that a collection is not an estimator
        """
        for cls in (list, dict, tuple, set):
            self.assertFalse(isestimator(cls))

        things = ['pepper', 'sauce', 'queen']
        self.assertFalse(isestimator(things))

    def test_visualizer_is_estimator(self):
        """
        Assert that a Visualizer is an estimator
        """
        self.assertTrue(is_estimator(Visualizer))
        self.assertTrue(is_estimator(Visualizer()))
        self.assertTrue(is_estimator(ScoreVisualizer))
        self.assertTrue(is_estimator(ScoreVisualizer(LinearRegression())))
        self.assertTrue(is_estimator(ModelVisualizer))
        self.assertTrue(is_estimator(ModelVisualizer(LogisticRegression())))

    ##////////////////////////////////////////////////////////////////////
    ## isregressor testing
    ##////////////////////////////////////////////////////////////////////

    def test_regressor_alias(self):
        """
        Assert is_regressor aliases isregressor
        """
        instance = LinearRegression()
        self.assertEqual(is_regressor(instance), isregressor(instance))

    def test_regressor_instance(self):
        """
        Test that is_regressor works for instances
        """

        # Test regressors are identified correctly
        regressors = (
            LassoCV,
            LinearRegression,
        )

        for model in regressors:
            instance = model()
            self.assertTrue(is_regressor(instance))

        # Test that non-regressors are identified correctly
        notregressors = (
            KMeans,
            PCA,
            LSHForest,
            LogisticRegression,
            RandomForestClassifier,
        )

        for model in notregressors:
            instance = model()
            self.assertFalse(is_regressor(instance))

    def test_regressor_class(self):
        """
        Test that is_regressor works for classes
        """

        # Test regressors are identified correctly
        regressors = (
            LassoCV,
            LinearRegression,
        )

        for klass in regressors:
            self.assertTrue(inspect.isclass(klass))
            self.assertTrue(is_regressor(klass))

        # Test that non-regressors are identified correctly
        notregressors = (
            KMeans,
            PCA,
            LSHForest,
            LogisticRegression,
            RandomForestClassifier,
        )

        for klass in notregressors:
            self.assertTrue(inspect.isclass(klass))
            self.assertFalse(is_regressor(klass))

    def test_regressor_pipeline(self):
        """
        Test that is_regressor works for pipelines
        """
        model = Pipeline([
            ('reduce_dim', PCA()),
            ('linreg', LinearRegression())
        ])

        self.assertTrue(is_regressor(model))

    def test_regressor_visualizer(self):
        """
        Test that is_regressor works on visualizers
        """
        model = ScoreVisualizer(LinearRegression())
        self.assertTrue(is_regressor(model))

    ##////////////////////////////////////////////////////////////////////
    ## isclassifier testing
    ##////////////////////////////////////////////////////////////////////

    def test_classifier_alias(self):
        """
        Assert is_classifier aliases isclassifier
        """
        instance = LogisticRegression()
        self.assertEqual(is_classifier(instance), isclassifier(instance))

    def test_classifier_instance(self):
        """
        Test that is_classifier works for instances
        """

        # Test classifiers are identified correctly
        classifiers = (
            LogisticRegression,
            RandomForestClassifier,
        )

        for model in classifiers:
            instance = model()
            self.assertTrue(is_classifier(instance))

        # Test that non-classifiers are identified correctly
        notclassifiers = (
            KMeans,
            PCA,
            LSHForest,
            LinearRegression,
            LassoCV,
        )

        for model in notclassifiers:
            instance = model()
            self.assertFalse(is_classifier(instance))

    def test_classifier_class(self):
        """
        Test that is_classifier works for classes
        """

        # Test classifiers are identified correctly
        classifiers = (
            RandomForestClassifier,
            LogisticRegression,
        )

        for klass in classifiers:
            self.assertTrue(inspect.isclass(klass))
            self.assertTrue(is_classifier(klass))

        # Test that non-regressors are identified correctly
        notclassifiers = (
            KMeans,
            PCA,
            LSHForest,
            LassoCV,
            LinearRegression,
        )

        for klass in notclassifiers:
            self.assertTrue(inspect.isclass(klass))
            self.assertFalse(is_classifier(klass))

    def test_classifier_pipeline(self):
        """
        Test that is_regressor works for pipelines
        """
        model = Pipeline([
            ('reduce_dim', PCA()),
            ('linreg', LogisticRegression())
        ])

        self.assertTrue(is_classifier(model))

    def test_classifier_visualizer(self):
        """
        Test that is_classifier works on visualizers
        """
        model = ScoreVisualizer(RandomForestClassifier())
        self.assertTrue(is_classifier(model))

##########################################################################
## Decorator Tests
##########################################################################

class DecoratorTests(unittest.TestCase):
    """
    Tests for the decorator utilities.
    """

    def test_docutil(self):
        """
        Test the docutil docstring copying methodology.
        """

        class Visualizer(object):

            def __init__(self):
                """
                This is the correct docstring.
                """
                pass


        def undecorated(*args, **kwargs):
            """
            This is an undecorated function string.
            """
            pass

        # Test the undecorated string to protect from magic
        self.assertEqual(
            undecorated.__doc__.strip(), "This is an undecorated function string."
        )

        # Decorate manually and test the newly decorated return function.
        decorated = docutil(Visualizer.__init__)(undecorated)
        self.assertEqual(
            decorated.__doc__.strip(), "This is the correct docstring."
        )

        # Assert that decoration modifies the original function.
        self.assertEqual(
            undecorated.__doc__.strip(), "This is the correct docstring."
        )

        @docutil(Visualizer.__init__)
        def sugar(*args, **kwargs):
            pass

        # Assert that syntactic sugar works as expected.
        self.assertEqual(
            sugar.__doc__.strip(), "This is the correct docstring."
        )


##########################################################################
## Execute Tests
##########################################################################

if __name__ == "__main__":
    unittest.main()
