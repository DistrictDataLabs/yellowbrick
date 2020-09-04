# tests.test_features.test_explained_variance
# Tests for the PCA explained variance visualizer
#
# Author:   Benjamin Bengfort
# Created:  Mon Feb 10 19:11:46 2020 -0500
#
# Copyright (C) 2019 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_explained_variance.py [] benjamin@bengfort.com $

"""
Tests for the PCA explained variance visualizer
"""

##########################################################################
## Imports
##########################################################################

from tests.base import VisualTestCase

from yellowbrick.datasets import load_credit
from yellowbrick.features.explained_variance import *


##########################################################################
## ExplainedVariance Tests
##########################################################################

class TextExplainedVariance(VisualTestCase):
    """
    Test the explained variance visualizer
    """

    def test_quick_method(self):
        """
        Test the explained variance quick method
        """
        X, _ = load_credit()
        oz = explained_variance(X)

        assert isinstance(oz, ExplainedVariance)
        self.assert_images_similar(oz)
