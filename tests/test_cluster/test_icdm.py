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
import matplotlib as mpl

from yellowbrick.cluster.icdm import *
from yellowbrick.exceptions import YellowbrickValueError
from sklearn.cluster import KMeans

from tests.base import VisualTestCase

try:
    import pandas as pd
except ImportError:
    pd = None


# Determine version of matplotlib
MPL_VERS_MAJ = int(mpl.__version__.split(".")[0])


##########################################################################
## InterclusterDistance Test Cases
##########################################################################

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

    @pytest.mark.skipif(MPL_VERS_MAJ >= 2, reason="test requires mpl earlier than 2.0.2")
    def test_legend_matplotlib_version(self, mock_toolkit):
        """
        ValueError is raised when matplotlib version is incorrect and legend=True
        """
        with pytst.raises(ImportError):
            from mpl_toolkits.axes_grid1 import inset_locator
            assert not inset_locator

        with pytest.raises(YellowbrickValueError, match="requires matplotlib 2.0.2"):
            InterclusterDistance(KMeans(), legend=True)

    @pytest.mark.skipif(MPL_VERS_MAJ >= 2, reason="test requires mpl earlier than 2.0.2")
    def test_no_legend_matplotlib_version(self, mock_toolkit):
        """
        No error is raised when matplotlib version is incorrect and legend=False
        """
        with pytst.raises(ImportError):
            from mpl_toolkits.axes_grid1 import inset_locator
            assert not inset_locator

        try:
            InterclusterDistance(KMeans(), legend=False)
        except YellowbrickValueError as e:
            self.fail(e)
