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

import sys
import pytest
import matplotlib as mpl

from yellowbrick.cluster.icdm import *
from yellowbrick.exceptions import YellowbrickValueError

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin, Dataset

from sklearn.datasets import make_blobs
from sklearn.cluster import Birch, AgglomerativeClustering
from sklearn.cluster import KMeans, AffinityPropagation, MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation as LDA

try:
    import pandas as pd
except ImportError:
    pd = None


# Determine version of matplotlib
MPL_VERS_MAJ = int(mpl.__version__.split(".")[0])


##########################################################################
## Fixtures
##########################################################################

@pytest.fixture(scope='class')
def blobs12(request):
    """
    Creates a fixture of 1000 instances in 12 clusters with 16 features.
    """
    X, y = make_blobs(
        centers=12, n_samples=1000, n_features=16, shuffle=True, random_state=2121
    )
    request.cls.blobs12 = Dataset(X, y)


@pytest.fixture(scope='class')
def blobs4(request):
    """
    Creates a fixture of 400 instances in 4 clusters with 16 features.
    """
    X, y = make_blobs(
        centers=4, n_samples=400, n_features=16, shuffle=True, random_state=1212
    )
    request.cls.blobs4 = Dataset(X, y)


def assert_fitted(oz):
    for param in ('cluster_centers_', 'embedded_centers_', 'scores_', 'fit_time_'):
        assert hasattr(oz, param)


def assert_not_fitted(oz):
    for param in ('embedded_centers_', 'scores_', 'fit_time_'):
        assert not hasattr(oz, param)



##########################################################################
## InterclusterDistance Test Cases
##########################################################################

@pytest.mark.usefixtures("blobs12", "blobs4")
class TestInterclusterDistance(VisualTestCase, DatasetMixin):
    """
    Test the InterclusterDistance visualizer
    """

    def test_only_valid_embeddings(self):
        """
        Should raise an exception on invalid embedding
        """
        # On init
        with pytest.raises(YellowbrickValueError, match="unknown embedding 'foo'"):
            InterclusterDistance(KMeans(), embedding='foo')

        # After init
        icdm = InterclusterDistance(KMeans())
        icdm.embedding = 'foo'
        with pytest.raises(YellowbrickValueError, match="unknown embedding 'foo'"):
            icdm.transformer

    def test_only_valid_scoring(self):
        """
        Should raise an exception on invalid scoring
        """
        # On init
        with pytest.raises(YellowbrickValueError, match="unknown scoring 'foo'"):
            InterclusterDistance(KMeans(), scoring='foo')

        # After init
        icdm = InterclusterDistance(KMeans())
        icdm.scoring = 'foo'
        with pytest.raises(YellowbrickValueError, match="unknown scoring method 'foo'"):
            icdm._score_clusters(None)

    def test_kmeans_mds(self):
        """
        Visual similarity with KMeans and MDS scaling
        """
        model = KMeans(9, random_state=38)
        oz = InterclusterDistance(model, random_state=83, embedding='mds')

        # Prefit assertions
        assert_not_fitted(oz)

        assert oz.fit(self.blobs12.X) is oz # Fit returns self

        # Postfit assertions
        assert_fitted(oz)
        assert oz.embedded_centers_.shape[0] == oz.scores_.shape[0]
        assert oz.embedded_centers_.shape[0] == oz.cluster_centers_.shape[0]
        assert len(oz._score_clusters(self.blobs12.X)) == 9
        assert len(oz._get_cluster_sizes()) == 9

        # Image similarity
        oz.finalize()
        tol = 4.9 if sys.platform == 'win32' else 1.0 # fails with RMSE 4.740 on AppVeyor
        self.assert_images_similar(oz, tol=tol)

    @pytest.mark.filterwarnings("ignore:the matrix subclass is not the recommended way")
    def test_affinity_tsne_no_legend(self):
        """
        Visual similarity with AffinityPropagation, TSNE scaling, and no legend
        """
        model = AffinityPropagation()
        oz = InterclusterDistance(
            model, random_state=763, embedding='tsne', legend=False
        )

        # Prefit assertions
        assert_not_fitted(oz)

        assert oz.fit(self.blobs4.X) is oz # Fit returns self

        # Postfit assertions
        assert_fitted(oz)
        assert oz.embedded_centers_.shape[0] == oz.scores_.shape[0]
        assert oz.embedded_centers_.shape[0] == oz.cluster_centers_.shape[0]

        # Image similarity
        oz.finalize()
        tol = 2.75 if sys.platform == 'win32' else 1.0 # fails with RMSE 2.687 on AppVeyor
        self.assert_images_similar(oz, tol=tol)


    @pytest.mark.skip(reason="LDA not implemented yet")
    def test_lda_mds(self):
        """
        Visual similarity with LDA and MDS scaling
        """
        model = LDA(9, random_state=6667)
        oz = InterclusterDistance(model, random_state=2332, embedding='mds')

        # Prefit assertions
        assert_not_fitted(oz)

        assert oz.fit(self.blobs12.X) is oz # Fit returns self

        # Postfit assertions
        assert_fitted(oz)
        assert oz.embedded_centers_.shape[0] == oz.scores_.shape[0]
        assert oz.embedded_centers_.shape[0] == oz.cluster_centers_.shape[0]
        assert len(oz._score_clusters(self.blobs12.X)) == 9
        assert len(oz._get_cluster_sizes()) == 9

        # Image similarity
        oz.finalize()
        self.assert_images_similar(oz, tol=1.0)

    @pytest.mark.skip(reason="agglomerative not implemented yet")
    @pytest.mark.filterwarnings("ignore:Using a non-tuple sequence")
    @pytest.mark.filterwarnings("ignore:the matrix subclass is not the recommended way")
    def test_birch_tsne(self):
        """
        Visual similarity with Birch and MDS scaling
        """
        oz = InterclusterDistance(Birch(n_clusters=9), random_state=83, embedding='mds')

        # Prefit assertions
        assert_not_fitted(oz)

        assert oz.fit(self.blobs12.X) is oz # Fit returns self

        # Postfit assertions
        assert_fitted(oz)
        assert oz.embedded_centers_.shape[0] == oz.scores_.shape[0]
        assert oz.embedded_centers_.shape[0] == oz.cluster_centers_.shape[0]
        assert len(oz._score_clusters(self.blobs12.X)) == 9
        assert len(oz._get_cluster_sizes()) == 9

        # Image similarity
        oz.finalize()
        self.assert_images_similar(oz, tol=1.0)

    @pytest.mark.skip(reason="agglomerative not implemented yet")
    def test_ward_mds_no_legend(self):
        """
        Visual similarity with Ward, TSNE scaling, and no legend
        """
        model = AgglomerativeClustering(n_clusters=9)
        oz = InterclusterDistance(
            model, random_state=83, embedding='tsne', legend=False
        )

        # Prefit assertions
        assert_not_fitted(oz)

        assert oz.fit(self.blobs12.X) is oz # Fit returns self

        # Postfit assertions
        assert_fitted(oz)
        assert oz.embedded_centers_.shape[0] == oz.scores_.shape[0]
        assert oz.embedded_centers_.shape[0] == oz.cluster_centers_.shape[0]
        assert len(oz._score_clusters(self.blobs12.X)) == 9
        assert len(oz._get_cluster_sizes()) == 9

        # Image similarity
        oz.finalize()
        self.assert_images_similar(oz, tol=1.0)

    def test_quick_method(self):
        """
        Test the quick method producing a valid visualization
        """
        model = MiniBatchKMeans(3, random_state=343)
        oz = intercluster_distance(model, self.blobs4.X, random_state=93, legend=False)
        assert isinstance(oz, InterclusterDistance)

        tol = 2.75 if sys.platform == 'win32' else 1.0 # fails with RMSE 2.631 on AppVeyor
        self.assert_images_similar(oz, tol=tol)

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
