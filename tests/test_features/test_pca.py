# tests.test_features.test_pca
# Tests for the PCA based feature visualizer.
#
# Author:   Carlo Morales <@cjmorale>
# Created:  Tue May 23 18:34:27 2017 -0400
#
# Copyright (C) 2017 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_pca.py [] cmorales@pacificmetrics.com $

"""
Tests for the PCA based feature visualizer.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import numpy.testing as npt

from tests.base import VisualTestCase
from yellowbrick.features.pca import *
from yellowbrick.exceptions import YellowbrickError


##########################################################################
##PCA Tests
##########################################################################

class PCADecompositionTests(VisualTestCase):
    """
    Test the PCADecomposition visualizer (scaled or non-scaled) for 2 and 3 dimensions.
    """
    def test_pca_decomposition(self):
        """
        Test the quick method PCADecomposition visualizer 2 dimensions scaled.
        """
        X = np.array(
                [[ 2.318, 2.727, 4.260, 7.212, 4.792],
                 [ 2.315, 2.726, 4.295, 7.140, 4.783,],
                 [ 2.315, 2.724, 4.260, 7.135, 4.779,],
                 [ 2.110, 3.609, 4.330, 7.985, 5.595,],
                 [ 2.110, 3.626, 4.330, 8.203, 5.621,],
                 [ 2.110, 3.620, 4.470, 8.210, 5.612,]]
            )

        y = np.array([1, 1, 0, 1, 0, 0])
        pca_decomposition(X=X, color=y, roj_dim=2, scale=True)

    def test_scale_true_2d(self):
        """
        Test the PCADecomposition visualizer 2 dimensions scaled.
        """
        X = np.array(
                [[ 2.318, 2.727, 4.260, 7.212, 4.792],
                 [ 2.315, 2.726, 4.295, 7.140, 4.783,],
                 [ 2.315, 2.724, 4.260, 7.135, 4.779,],
                 [ 2.110, 3.609, 4.330, 7.985, 5.595,],
                 [ 2.110, 3.626, 4.330, 8.203, 5.621,],
                 [ 2.110, 3.620, 4.470, 8.210, 5.612,]]
            )

        y = np.array([1, 1, 0, 1, 0, 0])

        params = {'scale': True, 'proj_dim': 2, 'col': y}
        visualizer = PCADecomposition(**params)
        visualizer.fit(X)
        pca_array = visualizer.transform(X)
        visualizer.poof()
        X_pca_decomp = np.array(
            [[-2.13928666, -0.07820484],
            [-2.0162836, 0.38910195],
            [-2.21597319, -0.05875371],
            [1.70792744, -0.6411635],
            [1.95978109, -0.71265712],
            [2.70383492, 1.10167722]]
            )
        npt.assert_array_almost_equal(pca_array, X_pca_decomp)


        params = {'scale': True, 'proj_dim': 2}
        visualizer = PCADecomposition(**params)
        visualizer.fit(X)
        pca_array = visualizer.transform(X)
        visualizer.poof()
        npt.assert_array_almost_equal(pca_array, X_pca_decomp)
        self.assert_images_similar(visualizer)


    def test_scale_false_2d(self):
        """
        Test the PCADecomposition visualizer 2 dimensions non-scaled.
        """
        X = np.array(
            [[2.318, 2.727, 4.260, 7.212, 4.792],
             [2.315, 2.726, 4.295, 7.140, 4.783, ],
             [2.315, 2.724, 4.260, 7.135, 4.779, ],
             [2.110, 3.609, 4.330, 7.985, 5.595, ],
             [2.110, 3.626, 4.330, 8.203, 5.621, ],
             [2.110, 3.620, 4.470, 8.210, 5.612, ]]
        )

        y = np.array([1, 1, 0, 1, 0, 0])

        params = {'scale': False, 'proj_dim': 2, 'col': y}
        visualizer = PCADecomposition(**params)
        visualizer.fit(X)
        pca_array = visualizer.transform(X)
        visualizer.poof()
        X_pca_decomp = np.array(
            [[-0.75173446, -0.02639709],
             [-0.79893433, -0.0028735],
             [-0.80765629, 0.01702425],
             [0.67843399, 0.11408186],
             [0.83702734, -0.00802634],
             [0.84286375, -0.09380918]]
        )
        npt.assert_array_almost_equal(pca_array, X_pca_decomp)

        params = {'scale': False, 'proj_dim': 2}
        visualizer = PCADecomposition(**params)
        visualizer.fit(X)
        pca_array = visualizer.transform(X)
        visualizer.poof()
        npt.assert_array_almost_equal(pca_array, X_pca_decomp)

    def test_scale_true_3d(self):
        """
        Test the PCADecomposition visualizer 3 dimensions scaled.
        """
        X = np.array(
                [[ 2.318, 2.727, 4.260, 7.212, 4.792],
                 [ 2.315, 2.726, 4.295, 7.140, 4.783,],
                 [ 2.315, 2.724, 4.260, 7.135, 4.779,],
                 [ 2.110, 3.609, 4.330, 7.985, 5.595,],
                 [ 2.110, 3.626, 4.330, 8.203, 5.621,],
                 [ 2.110, 3.620, 4.470, 8.210, 5.612,]]
            )

        y = np.array([1, 1, 0, 1, 0, 0])

        params = {'scale': True, 'proj_dim': 3, 'col': y}
        visualizer = PCADecomposition(**params)
        visualizer.fit(X)
        pca_array = visualizer.transform(X)
        visualizer.poof()
        X_pca_decomp = np.array(
            [[-2.13928666, -0.07820484, -0.11005612],
            [-2.0162836, 0.38910195, 0.06538246],
            [-2.21597319, -0.05875371, 0.03015729],
            [1.70792744, -0.6411635, 0.20001772],
            [1.95978109, -0.71265712, -0.16553243],
            [2.70383492, 1.10167722,  -0.01996893]]
            )
        npt.assert_array_almost_equal(pca_array, X_pca_decomp)

        params = {'scale': True, 'proj_dim': 3}
        visualizer = PCADecomposition(**params)
        visualizer.fit(X)
        pca_array = visualizer.transform(X)
        visualizer.poof()

        npt.assert_array_almost_equal(pca_array, X_pca_decomp)

    def test_scale_false_3d(self):
        """
        Test the PCADecomposition visualizer 3 dimensions non-scaled.
        """
        X = np.array(
                [[ 2.318, 2.727, 4.260, 7.212, 4.792],
                 [ 2.315, 2.726, 4.295, 7.140, 4.783,],
                 [ 2.315, 2.724, 4.260, 7.135, 4.779,],
                 [ 2.110, 3.609, 4.330, 7.985, 5.595,],
                 [ 2.110, 3.626, 4.330, 8.203, 5.621,],
                 [ 2.110, 3.620, 4.470, 8.210, 5.612,]]
            )

        y = np.array([1, 1, 0, 1, 0, 0])

        params = {'scale': False, 'proj_dim': 3, 'col': y}
        visualizer = PCADecomposition(**params)
        visualizer.fit(X)
        pca_array = visualizer.transform(X)
        visualizer.poof()
        X_pca_decomp = np.array(
                [[ -7.51734458e-01,  -2.63970949e-02,   3.23270821e-02],
                 [ -7.98934328e-01,  -2.87350350e-03,  -2.86110098e-02],
                 [ -8.07656292e-01,   1.70242492e-02,  -4.98720042e-04],
                 [  6.78433990e-01,   1.14081863e-01,  -2.51825210e-02],
                 [  8.37027339e-01,  -8.02633755e-03,   6.65986453e-02],
                 [  8.42863750e-01,  -9.38091760e-02,  -4.46334766e-02]]
            )
        npt.assert_array_almost_equal(pca_array, X_pca_decomp)


        params = {'scale': False, 'proj_dim': 3}
        visualizer = PCADecomposition(**params)
        visualizer.fit(X)
        pca_array = visualizer.transform(X)
        npt.assert_array_almost_equal(pca_array, X_pca_decomp)

    def test_scale_true_4d_execption(self):
        """
        Test the PCADecomposition visualizer 4 dimensions scaled (catch YellowbrickError).
        """
        params = {'scale': True, 'center': False, 'proj_dim': 4}
        with self.assertRaisesRegexp(YellowbrickError, "proj_dim object is not 2 or 3"):
            PCADecomposition(**params)

    def test_scale_true_3d_execption(self):
        """
        Test the PCADecomposition visualizer 3 dims scaled on 2 dim data set (catch ValueError).
        """
        X = np.array(
            [[2.318, 2.727],
             [2.315, 2.726],
             [2.315, 2.724],
             [2.110, 3.609],
             [2.110, 3.626],
             [2.110, 3.620]]
        )

        y = np.array([1, 0])

        params = {'scale': True, 'center': False, 'proj_dim': 3, 'col': y}


        with self.assertRaisesRegexp(ValueError, "n_components=3 must be between 0 and n_features"):
            pca = PCADecomposition(**params)
            pca.fit(X)
