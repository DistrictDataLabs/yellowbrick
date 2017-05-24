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
from sklearn.datasets import load_digits
from sklearn.model_selection import ShuffleSplit
from yellowbrick.classifier.learning_curve import LearningCurveVisualizer
from yellowbrick.classifier.learning_curve import learning_curve_plot
from yellowbrick.exceptions import YellowbrickError
from tests.dataset import DatasetMixin

##########################################################################
## LearningCurveTests Test Cases
##########################################################################

class LearningCurveTests(VisualTestCase, DatasetMixin):

    def setUp(self):
        self.occupancy = self.load_data('occupancy')
        

    def tearDown(self):
        self.occupancy = None
        X = None
        y = None

    def test_learning_curve_comprehensive(self):
        """
        Test learning curve with all parameters.
        """

        X = self.occupancy[[
                "temperature", "relative_humidity", "light", "C02", "humidity"
            ]]

        y = self.occupancy['occupancy'].astype(int)

        X = X.view((float, len(X.dtype.names)))

        try:
            visualizer = LearningCurveVisualizer(LinearSVC(), train_sizes=np.linspace(.1, 1.0, 5), 
                cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0), 
                n_jobs=4)
            visualizer.fit(X, y)
            visualizer.poof()
        except Exception as e:
            self.fail("error during learning curve: {}".format(e))

    def test_learning_curve_model_only(self):
        """
        Test learning curve with inputting model only.
        """

        X = self.occupancy[[
                "temperature", "relative_humidity", "light", "C02", "humidity"
            ]]

        y = self.occupancy['occupancy'].astype(int)

        X = X.view((float, len(X.dtype.names)))
        
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

        X = self.occupancy[[
                "temperature", "relative_humidity", "light", "C02", "humidity"
            ]]

        y = self.occupancy['occupancy'].astype(int)

        X = X.view((float, len(X.dtype.names)))
        
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

        X = self.occupancy[[
                "temperature", "relative_humidity", "light", "C02", "humidity"
            ]]

        y = self.occupancy['occupancy'].astype(int)

        X = X.view((float, len(X.dtype.names)))
        
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

        X = self.occupancy[[
                "temperature", "relative_humidity", "light", "C02", "humidity"
            ]]

        y = self.occupancy['occupancy'].astype(int)

        X = X.view((float, len(X.dtype.names)))
        
        with self.assertRaises(YellowbrickError):
            visualizer = LearningCurveVisualizer(LinearSVC(), 
                train_sizes=10000,
                cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0))
            visualizer.fit(X, y)
            visualizer.poof()
        
