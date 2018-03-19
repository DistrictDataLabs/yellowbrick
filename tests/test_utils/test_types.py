# tests.test_utils.test_types
# Very difficult test library for type detection and flexibility.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri May 19 10:58:32 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_types.py [79cd8cf] benjamin@bengfort.com $

"""
Very difficult test library for type detection and flexibility.
"""

##########################################################################
## Imports
##########################################################################

import inspect
import unittest

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import AffinityPropagation, Birch

from yellowbrick.utils.types import *
from yellowbrick.base import Visualizer, ScoreVisualizer, ModelVisualizer


##########################################################################
## Model Utility Tests
##########################################################################

class ModelUtilityTests(unittest.TestCase):

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
            NearestNeighbors(),
            PCA(),
            RidgeCV(),
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
            NearestNeighbors,
            PCA,
            RidgeCV,
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
            RidgeCV,
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
            NearestNeighbors,
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
            RidgeCV,
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
            NearestNeighbors,
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
            NearestNeighbors,
            LinearRegression,
            RidgeCV,
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
            NearestNeighbors,
            RidgeCV,
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

    ##////////////////////////////////////////////////////////////////////
    ## isclusterer testing
    ##////////////////////////////////////////////////////////////////////

    def test_clusterer_alias(self):
        """
        Assert is_clusterer aliases isclusterer
        """
        instance = KMeans()
        self.assertEqual(is_clusterer(instance), isclusterer(instance))

    def test_clusterer_instance(self):
        """
        Test that is_clusterer works for instances
        """

        # Test clusterers are identified correctly
        clusterers = (
            KMeans,
            MiniBatchKMeans,
            AffinityPropagation,
            Birch
        )

        for model in clusterers:
            instance = model()
            self.assertTrue(is_clusterer(instance))

        # Test that non-clusterers are identified correctly
        notclusterers = (
            RidgeCV,
            LassoCV,
            LinearRegression,
            PCA,
            NearestNeighbors,
            LogisticRegression,
            RandomForestClassifier,
        )

        for model in notclusterers:
            instance = model()
            self.assertFalse(is_clusterer(instance))

    def test_clusterer_class(self):
        """
        Test that is_clusterer works for classes
        """

        # Test clusterers are identified correctly
        clusterers = (
            KMeans,
            MiniBatchKMeans,
            AffinityPropagation,
            Birch
        )

        for klass in clusterers:
            self.assertTrue(inspect.isclass(klass))
            self.assertTrue(is_clusterer(klass))

        # Test that non-clusterers are identified correctly
        notclusterers = (
            RidgeCV,
            LassoCV,
            LinearRegression,
            PCA,
            NearestNeighbors,
            LogisticRegression,
            RandomForestClassifier,
        )

        for klass in notclusterers:
            self.assertTrue(inspect.isclass(klass))
            self.assertFalse(is_clusterer(klass))

    def test_clusterer_pipeline(self):
        """
        Test that is_clusterer works for pipelines
        """
        model = Pipeline([
            ('reduce_dim', PCA()),
            ('kmeans', KMeans())
        ])

        self.assertTrue(is_clusterer(model))

    def test_clusterer_visualizer(self):
        """
        Test that is_clusterer works on visualizers
        """
        model = ScoreVisualizer(KMeans())
        self.assertTrue(is_clusterer(model))


class StructuredArrayTests(unittest.TestCase):

    def test_isstructuredarray_true(self):
        x = np.array([(1,2.,'Hello'), (2,3.,"World")], dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'S10')])
        self.assertTrue(isstructuredarray(x))

    def test_isstructuredarray_false(self):
        x = np.array([[1,2,3], [1,2,3]])
        self.assertFalse(isstructuredarray(x))

    def test_isstructuredarray_list(self):
        x = [[1,2,3], [1,2,3]]
        self.assertFalse(isstructuredarray(x))

##########################################################################
## Execute Tests
##########################################################################

if __name__ == "__main__":
    unittest.main()
