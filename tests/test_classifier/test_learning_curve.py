# tests.test_classifier.test_learning_curve
# Tests for the LearningCurveVisualizer
#
# Author:   Jason Keung <jason.s.keung@gmail.com>
# Created:  Tues May 23 11:45:00 2017 -0400
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_learning_curve.py jason.s.keung@gmail.com $

"""
Tests for the LearningCurveVisualizer
"""

##########################################################################
## Imports
##########################################################################

import unittest
import numpy as np

from ..base import VisualTestCase

from sklearn.svm import LinearSVC
from sklearn.model_selection import ShuffleSplit
from yellowbrick.classifier.learning_curve import LearningCurveVisualizer
from yellowbrick.classifier.learning_curve import learning_curve_plot
from yellowbrick.exceptions import YellowbrickError
from tests.dataset import DatasetMixin

##########################################################################
# Data
##########################################################################

np.random.seed(0)
X = np.random.random((1000, 5))
y = np.random.random_integers(0, 1, (1000))

##########################################################################
## LearningCurveTests Test Cases
##########################################################################

class LearningCurveTests(VisualTestCase, DatasetMixin):

    def setUp(self):
        super(LearningCurveTests, self).setUp()

    @unittest.skip("Image not close")
    def test_learning_curve_comprehensive(self):
        """
        Test learning curve with all parameters with visual unit test.
        """

        try:
            visualizer = LearningCurveVisualizer(LinearSVC(random_state=0), train_sizes=np.linspace(.1, 1.0, 5),
                cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0),
                n_jobs=4)
            visualizer.fit(X, y)
            visualizer.poof()
        except Exception as e:
            self.fail("error during learning curve: {}".format(e))

        self.assert_images_similar(visualizer)

    def test_learning_curve_model_only(self):
        """
        Test learning curve with inputting model only.
        """

        try:
            visualizer = LearningCurveVisualizer(LinearSVC())
            visualizer.fit(X, y)
            visualizer.poof()
        except Exception as e:
            self.fail("error during learning curve: {}".format(e))

    def test_learning_curve_model_cv_only(self):
        """
        Test learning curve with inputting model and cv only.
        """

        try:
            visualizer = LearningCurveVisualizer(LinearSVC(),
                cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0))
            visualizer.fit(X, y)
            visualizer.poof()
        except Exception as e:
            self.fail("error during learning curve: {}".format(e))

    def test_learning_curve_model_trainsize_cv_only(self):
        """
        Test learning curve with inputting model, training size, and cv only.
        """

        try:
            visualizer = LearningCurveVisualizer(LinearSVC(),
                train_sizes=np.linspace(.1, 1.0, 5),
                cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0))
            visualizer.fit(X, y)
            visualizer.poof()
        except Exception as e:
            self.fail("error during learning curve: {}".format(e))

    def test_learning_curve_bad_trainsize(self):
        """
        Test learning curve with bad input for training size.
        """

        with self.assertRaises(YellowbrickError):
            visualizer = LearningCurveVisualizer(LinearSVC(),
                train_sizes=10000,
                cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0))
            visualizer.fit(X, y)
            visualizer.poof()

    def test_learning_curve_quick_method(self):
        """
        Test the learning curve quick method acts as expected
        """
        try:
            learning_curve_plot(
                X, y,
                LinearSVC(random_state=0),
                train_sizes=np.linspace(.1, 1.0, 5),
                cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0),
                n_jobs=4
            )
        except Exception as e:
            self.fail("error during learning curve: {}".format(e))

        # TODO: assert images are similar
        # self.assert_images_similar(visualizer)
