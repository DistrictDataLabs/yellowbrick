##########################################################################
## Imports
##########################################################################

import unittest
import numpy as np
import numpy.testing as npt

from tests.dataset import DatasetMixin
from yellowbrick.features.pca import PCA2D

##########################################################################
## RadViz Base Tests
##########################################################################

class PCA2DTests(unittest.TestCase, DatasetMixin):

    def test_scale_t_center_t(self, scale=True, center=True):

        X = np.array(
                [[ 2.318, 2.727, 4.260, 7.212, 4.792],
                 [ 2.315, 2.726, 4.295, 7.140, 4.783,],
                 [ 2.315, 2.724, 4.260, 7.135, 4.779,],
                 [ 2.110, 3.609, 4.330, 7.985, 5.595,],
                 [ 2.110, 3.626, 4.330, 8.203, 5.621,],
                 [ 2.110, 3.620, 4.470, 8.210, 5.612,]]
            )

        y = np.array([1, 1, 0, 1, 0, 0])

        params = {'scale': scale, 'center': center, 'col': y}
        visualizer = PCA2D(**params)
        visualizer.fit(X)
        pca_array = visualizer.transform(X)

        X_pca_decomp = np.array(
            [[-2.13928666, -0.07820484],
            [-2.0162836, 0.38910195],
            [-2.21597319, -0.05875371],
            [1.70792744, -0.6411635],
            [1.95978109, -0.71265712],
            [2.70383492, 1.10167722]]
            )
        npt.assert_array_almost_equal(pca_array, X_pca_decomp)

    def test_scale_f_center_f(self, scale=False, center=False):
        X = np.array(
            [[2.318, 2.727, 4.260, 7.212, 4.792],
             [2.315, 2.726, 4.295, 7.140, 4.783, ],
             [2.315, 2.724, 4.260, 7.135, 4.779, ],
             [2.110, 3.609, 4.330, 7.985, 5.595, ],
             [2.110, 3.626, 4.330, 8.203, 5.621, ],
             [2.110, 3.620, 4.470, 8.210, 5.612, ]]
        )

        y = np.array([1, 1, 0, 1, 0, 0])

        params = {'scale': scale, 'center': center, 'col': y}
        visualizer = PCA2D(**params)
        visualizer.fit(X)
        pca_array = visualizer.transform(X)

        X_pca_decomp = np.array(
            [[-0.75173446, -0.02639709],
             [-0.79893433, -0.0028735],
             [-0.80765629, 0.01702425],
             [0.67843399, 0.11408186],
             [0.83702734, -0.00802634],
             [0.84286375, -0.09380918]]
        )
        npt.assert_array_almost_equal(pca_array, X_pca_decomp)