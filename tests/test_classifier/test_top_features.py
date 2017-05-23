# tests.test_classifier.test_top_features
# Testing top N features of a classifier
#
# Author:   Elaine Ayo <@ayota>
# Created:  Tue May 23 14:10:42 2017 -0400
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_top_features.py [] $

"""
Testing for the top N features visualizer

TODO:
-   Cover all estimators outlined in issue #58
"""

import numpy as np

from yellowbrick.classifier.top_features import *
from tests.base import VisualTestCase
from tests.dataset import DatasetMixin

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

# some other model with no coef or feature importance param

##########################################################################
## Data
##########################################################################

X = np.array(
        [[ 2.318, 2.727, 4.260, 7.212, 4.792],
         [ 2.315, 2.726, 4.295, 7.140, 4.783,],
         [ 2.315, 2.724, 4.260, 7.135, 4.779,],
         [ 2.110, 3.609, 4.330, 7.985, 5.595,],
         [ 2.110, 3.626, 4.330, 8.203, 5.621,],
         [ 2.110, 3.620, 4.470, 8.210, 5.612,]]
    )

y = np.array([1, 1, 0, 1, 0, 0])

##########################################################################
##  Test for Top Features Visualizer
##########################################################################

class TopFeaturesTests(VisualTestCase, DatasetMixin):

    def setUp(self):
        corpus = self.load_data('hobbies')
        data = corpus.data
        self.target = corpus.target
        vectorizer = CountVectorizer(stop_words='english')
        vectorizer.fit(data)
        self.X_train = vectorizer.transform(data)
        self.features = vectorizer.get_feature_names()

    def tearDown(self):
        self.target = None
        self.X_train = None
        self.features = None

    def test_basic(self):
        """
        Test basic usage on hobbies dataset.
        """
        try:
            model = LinearSVC()
            visualizer = TopFeaturesVisualizer(model, N=5, features=self.features)
            visualizer.fit(self.X_train, self.target)
            visualizer.score(class_label='sports')
            visualizer.poof()
        except Exception as e:
            self.fail('Basic visualizer test failed.')

    def test_too_many_features(self):
        """
        Make sure replaces N with number of features in data
        when N is greater than number of features.
        """
        try:
            model = LinearSVC()
            visualizer = TopFeaturesVisualizer(model, N=10)
            visualizer.fit(X, y)
            visualizer.score(class_label=0)
            visualizer.poof()
        except Exception as e:
            self.fail('Test of N > number of features failed.')

    # def test_warn_features(self):
    #     """
    #     Raise warning if feature names not included.
    #     """
    #     pass
    #
    # def test_error_nolabel(self):
    #     """
    #     Raise error is no class label passed.
    #     """
    #     pass

    # def test_param_coef(self):
    #     """
    #     Make sure estimators with coef params work properly.
    #     """
    #     pass
    #
    # def test_param_feature_importance(self):
    #     """
    #     Make sure estimators with feature importance params raise errors.
    #     """
    #     pass
    #
    # def test_param_none(self):
    #     """
    #     Make sure estimators without coef or feature importance params raise
    #     the correct errors.
    #     """
    #     pass