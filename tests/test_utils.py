# tests.test_utils
# Test the export module - to generate a corpus for machine learning.
#
# Author:   Jason Keung <jason.s.keung@gmail.com>
#           Patrick O'Melveny <pvomelveny@gmail.com>
# Created:  Thurs Jun 2 15:33:18 2016 -0500
#
# For license information, see LICENSE.txt
#


"""
Test the utils module - to generate a corpus for machine learning.
"""

##########################################################################
## Imports
##########################################################################

from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LSHForest
from sklearn.pipeline import Pipeline
import unittest

from yellowbrick.utils import get_model_name


class ModelNameTests(unittest.TestCase):

    def test_real_model(self):
        model1 = LassoCV()
        model2 = LSHForest()
        self.assertEqual(get_model_name(model1), 'LassoCV')
        self.assertEqual(get_model_name(model2), 'LSHForest')

    def test_pipeline(self):
        pipeline = Pipeline([('reduce_dim', PCA()),
                             ('linreg', LinearRegression())])
        self.assertEqual(get_model_name(pipeline), 'LinearRegression')

    def test_int_input(self):

        self.assertRaises(TypeError, get_model_name, 1)

    def test_str_input(self):

        self.assertRaises(TypeError, get_model_name, 'helloworld')


if __name__ == "__main__":
    unittest.main()
