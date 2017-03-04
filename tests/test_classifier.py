# tests.test_classifier
# Tests for the classifiers module.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  Sat Oct 8 16:30:39 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
"""
Testing for the Classification Score Visualizers
"""

##########################################################################
## Imports
##########################################################################

import unittest

from tests.base import VisualTestCase
from yellowbrick.classifier import *

from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

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
##  Test for ROC-AUC Curve
##########################################################################

class ROCAUCTests(VisualTestCase):

    def test_roc_auc(self):
        """
        Assert no errors occur during ROC-AUC integration
        """
        model = LinearSVC()
        model.fit(X,y)
        visualizer = ROCAUC(model, classes=["A", "B"])
        visualizer.score(X,y)


##########################################################################
##  Test for Classification Report
##########################################################################

class ClassificationReportTests(VisualTestCase):

    def test_class_report(self):
        """
        Assert no errors occur during classification report integration
        """
        model = LinearSVC()
        model.fit(X,y)
        visualizer = ClassificationReport(model, classes=["A", "B"])
        visualizer.score(X,y)

class ConfusionMatrixTests(VisualTestCase):
    def __init__(self, *args, **kwargs):
        super(ConfusionMatrixTests, self).__init__(*args, **kwargs)
        #Use the same data for all the tests
        self.digits = load_digits()

        X = self.digits.data
        y = self.digits.target
        
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state=11)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def test_confusion_matrix(self):
        model = LogisticRegression()
        cm = ConfusionMatrix(model, classes=[0,1,2,3,4,5,6,7,8,9])
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)

    def test_no_classes_provided(self):
        model = LogisticRegression()
        cm = ConfusionMatrix(model)
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)

    def test_raw_count_mode(self):
        model = LogisticRegression()
        cm = ConfusionMatrix(model)
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test, percent=False)

    def test_zoomed_in(self):
        model = LogisticRegression()
        cm = ConfusionMatrix(model, classes=[0,1,2])
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)

    def test_extra_classes(self):
        model = LogisticRegression()
        cm = ConfusionMatrix(model, classes=[0,1,2,11])
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)
        self.assertTrue(cm.selected_class_counts[3]==0)

    def test_one_class(self):
        model = LogisticRegression()
        cm = ConfusionMatrix(model, classes=[0])
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)