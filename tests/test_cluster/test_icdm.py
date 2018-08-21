# tests.test_cluster.test_icdm
# Tests for the intercluster distance map visualizer.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Tue Aug 21 11:57:44 2018 -0400
#
# ID: test_icdm.py [] benjamin@bengfort.com $

"""
Tests for the intercluster distance map visualizer.
"""

##########################################################################
## Imports
##########################################################################

import pytest

from yellowbrick.cluster.icdm import *
from yellowbrick.exceptions import YellowbrickValueError
from sklearn.cluster import KMeans

from tests.base import VisualTestCase


class TestInterclusterDistance(VisualTestCase):
    """
    Test the InterclusterDistance visualizer
    """

    def test_only_valid_embeddings(self):
        """
        should raise an exception on invalid embedding
        """
        with pytest.raises(YellowbrickValueError, match="unknown embedding 'foo'"):
            InterclusterDistance(KMeans(), embedding='foo')

    def test_only_valid_scoring(self):
        """
        should raise an exception on invalid scoring
        """
        with pytest.raises(YellowbrickValueError, match="unknown scoring 'foo'"):
            InterclusterDistance(KMeans(), scoring='foo')
