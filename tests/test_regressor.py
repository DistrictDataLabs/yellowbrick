# tests.test_regressor
# Ensure that the regressor visualizations work.
#
# Author:   Rebecca Bilbro <rbilbro@districtdatalabs.com>
# Created:  Sat Oct 8 16:30:39 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#

"""
Ensure that the regressor visualizations work.
"""

##########################################################################
## Imports
##########################################################################

import unittest
import numpy as np

from tests.base import VisualTestCase
from yellowbrick.regressor import *

from sklearn.svm import SVR
from sklearn import cross_validation as cv
from sklearn.cross_validation import train_test_split as tts

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

y = np.array([0.23, .33, .31, .3, .24, .32, 0.23, .33, .31, .3, .24, .32])


##########################################################################
## Prediction error test case
##########################################################################

class PredictionErrorTests(VisualTestCase):

    def test_pred_error(self):
        """
        Assert no errors occur during Prediction Error Plots integration
        """
        model = SVR()
        model.fit(X, y)
        visualizer = PredictionError(model)
        visualizer.score(X, y)

##########################################################################
## Residuals Plots test case
##########################################################################

class ResidualsPlotTests(VisualTestCase):

    def test_resid_plots(self):
        """
        Assert no errors occur during Residual Plots integration
        """
        model = SVR()
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.5)
        model.fit(X_train, y_train)
        visualizer = ResidualsPlot(model)
        visualizer.score(X_test, y_test)
