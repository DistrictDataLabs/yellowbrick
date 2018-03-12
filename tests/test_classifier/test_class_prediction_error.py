# tests.test_classifier.test_class_prediction_error
# Testing for the ClassPredictionError visualizer
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Author:   Larry Gray
# Created:  Tue May 23 13:41:55 2017 -0700
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_rocauc.py [] benjamin@bengfort.com $

"""                                                                                                                  Testing for the ClassPredictionError visualizer
"""

##########################################################################
## Imports
##########################################################################

import pytest
import numpy as np

from yellowbrick.classifier.class_balance import *

from tests.base import VisualTestCase
from sklearn.svm import LinearSVC

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
##  Tests
##########################################################################

class ClassPredictionErrorTests(VisualTestCase):

    @pytest.mark.skip(reason="not implemented yet")
    def test_class_report(self):
        """
        Assert no errors occur during class prediction error integration
        """
        model = LinearSVC()
        model.fit(X,y)
        visualizer = ClassPredictionError(model, classes=["A", "B"])
        visualizer.score(X,y)
        self.assert_images_similar(visualizer)

    @pytest.mark.skip(reason="not implemented yet")
    def test_no_classes_provided(self):
        """
        Assert no errors when no classes are provided
        """
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_class_mismatch(self):
        """
        Test mismatch between the number of classes and indices
        """
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_class_type(self):
        """
        Test class must be either binary or multiclass type
        """
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_class_prediction_error_quickmethod(self):
        """
        Test the ClassPreditionError quickmethod
        """
        pass
