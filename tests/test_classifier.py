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
