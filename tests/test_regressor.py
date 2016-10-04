# tests.test_regressor
# Ensure that the regressor visualizations work.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Jun 03 14:20:02 2016 -0700
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_regressor.py [be63645] benjamin@bengfort.com $

"""
Ensure that the regressor visualizations work.
"""

##########################################################################
## Imports
##########################################################################

import unittest

from yellowbrick.regressor import *
from yellowbrick.utils import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

##########################################################################
## Prediction error test case
##########################################################################

class PredictionErrorTests(unittest.TestCase):

    @unittest.skip("Needs to be configured for the new API")
    def test_init_pe_viz(self):
        """
        Ensure that both a single model and multiple models can be rendered
        """
        viz = PredictionError([RandomForestRegressor(), SVR()])
        self.assertEqual(len(viz.models), 2)

        viz = PredictionError(SVR())
        self.assertEqual(len(viz.models), 1)

    @unittest.skip("Needs to be configured for the new API")
    def test_init_pe_names(self):
        """
        Ensure that model names are correctly extracted
        """
        viz = PredictionError([RandomForestRegressor(), SVR()])
        self.assertEqual(viz.names, ["RandomForestRegressor", "SVR"])

        viz = PredictionError(SVR())
        self.assertEqual(viz.names, ["SVR"])
