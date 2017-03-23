# tests.test_neighbors.test_knn
# Ensure that the KNN boundary visualizations work.
#
# Author:   Author:   Nathan Danielsen <nathan.danielsen@gmail.com>
# Created:  Sun Mar 19 13:01:29 2017 -0400
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_knn.py [] nathan.danielsen@gmail.com $

"""
Ensure that the KNN boundary visualizations work.
"""

##########################################################################
## Imports
##########################################################################

import unittest
import numpy as np

from tests.base import VisualTestCase
from yellowbrick.neighbors.knn import *

from sklearn import neighbors


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

y = np.array([1, 2, 1, 2, 1, 0, 0, 1, 3, 1, 3, 2])

##########################################################################
## Residuals Plots test case
##########################################################################

class KnnDecisionBoundariesVisualizerTest(VisualTestCase):

    def test_knn_decision_bounardies(self):
        """
        Assert no errors occur during KnnDecisionBoundariesVisualizer integration
        """
        X_two_cols =  X[:,:2]
        model = neighbors.KNeighborsClassifier(15)
        visualizer = KnnDecisionBoundariesVisualizer(model)#, classes=[1,2,3,4])
        visualizer.fit(X_two_cols, y=y)
        visualizer.draw(X_two_cols, y=y)
        # visualizer.poof()
