# tests.test_cluster.test_elbow
# Tests for the KElbowVisualizer
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Mar 23 22:30:19 2017 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: test_elbow.py [5a370c8] benjamin@bengfort.com $

"""
Tests for the KElbowVisualizer
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest
import numpy as np
import matplotlib.pyplot as plt

from ..base import VisualTestCase
from ..dataset import DatasetMixin

from scipy.sparse import csc_matrix, csr_matrix
from numpy.testing.utils import assert_array_almost_equal

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from yellowbrick.cluster.elbow import distortion_score
from yellowbrick.cluster.elbow import KElbowVisualizer
from yellowbrick.exceptions import YellowbrickValueError

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## K-Elbow Helpers Test Cases
##########################################################################

X = np.array(
      [[-0.40020753, -4.67055317, -0.27191127, -1.49156318],
       [ 0.37143349, -4.89391622, -1.23893945,  0.48318165],
       [ 8.625142  , -1.2372284 ,  1.39301471,  4.3394457 ],
       [ 7.65803596, -2.21017215,  1.99175714,  3.71004654],
       [ 0.89319875, -5.37152317,  1.50313598,  1.95284886],
       [ 2.68362166, -5.78810913, -0.41233406,  1.94638989],
       [ 7.63541182, -1.99606076,  0.9241231 ,  4.53478238],
       [ 9.04699415, -0.74540679,  0.98042851,  5.99569071],
       [ 1.02552122, -5.73874278, -1.74804915, -0.07831216],
       [ 7.18135665, -3.49473178,  1.14300963,  4.46065816],
       [ 0.58812902, -4.66559815, -0.72831685,  1.40171779],
       [ 1.48620862, -5.9963108 ,  0.19145963, -1.11369256],
       [ 7.6625556 , -1.21328083,  2.06361094,  6.2643551 ],
       [ 9.45050727, -1.36536078,  1.31154384,  3.89103468],
       [ 6.88203724, -1.62040255,  3.89961049,  2.12865388],
       [ 5.60842705, -2.10693356,  1.93328514,  3.90825432],
       [ 2.35150936, -6.62836131, -1.84278374,  0.51540886],
       [ 1.17446451, -5.62506058, -2.18420699,  1.21385128]]
)

y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0])


class TestKElbowHelper(object):
    """
    Helper functions for K-Elbow Visualizer
    """

    def test_distortion_score(self):
        """
        Test the distortion score metric function
        """
        score = distortion_score(X, y)
        assert score == 7.6777850157143783

    @pytest.mark.parametrize("Xs", [
        csc_matrix(X), csr_matrix(X),
    ], ids=["csc", "csr"])
    def test_distortion_score_sparse_matrix_input(self, Xs):
        """
        Test the distortion score metric on a sparse array
        """
        score = distortion_score(Xs, y)
        assert score == pytest.approx(7.6777850157143783)

    @pytest.mark.skipif(pd is None, reason="pandas is required")
    def test_distortion_score_pandas_input(self):
        """
        Test the distortion score metric on pandas DataFrame and Series
        """
        df = pd.DataFrame(X)
        s = pd.Series(y)

        score = distortion_score(df, s)
        assert score == pytest.approx(7.6777850157143783)


##########################################################################
## KElbowVisualizer Test Cases
##########################################################################

class TestKElbowVisualizer(VisualTestCase, DatasetMixin):
    """
    K-Elbow Visualizer Tests
    """

    @pytest.mark.xfail(reason="images not close due to timing lines")
    def test_integrated_kmeans_elbow(self):
        """
        Test no exceptions for kmeans k-elbow visualizer on blobs dataset
        """
        # NOTE #182: cannot use occupancy dataset because of memory usage

        # Generate a blobs data set
        X,y = make_blobs(
            n_samples=1000, n_features=12, centers=6,
            shuffle=True, random_state=42
        )

        try:
            _, ax = plt.subplots()

            visualizer = KElbowVisualizer(KMeans(random_state=42), k=4, ax=ax)
            visualizer.fit(X)
            visualizer.poof()

            self.assert_images_similar(visualizer)
        except Exception as e:
            pytest.fail("error during k-elbow: {}".format(e))

    @pytest.mark.xfail(reason="images not close due to timing lines")
    def test_integrated_mini_batch_kmeans_elbow(self):
        """
        Test no exceptions for mini-batch kmeans k-elbow visualizer
        """
        # NOTE #182: cannot use occupancy dataset because of memory usage

        # Generate a blobs data set
        X,y = make_blobs(
            n_samples=1000, n_features=12, centers=6, shuffle=True, random_state=42
        )

        try:
            _, ax = plt.subplots()

            visualizer = KElbowVisualizer(
                MiniBatchKMeans(random_state=42), k=4, ax=ax
            )
            visualizer.fit(X)
            visualizer.poof()

            self.assert_images_similar(visualizer)
        except Exception as e:
            pytest.fail("error during k-elbow: {}".format(e))

    @pytest.mark.skip(reason="takes over 20 seconds to run")
    def test_topic_modeling_k_means(self):
        """
        Test topic modeling k-means on the hobbies corpus
        """
        corpus = self.load_corpus("hobbies")

        tfidf  = TfidfVectorizer()
        docs   = tfidf.fit_transform(corpus.data)
        visualizer = KElbowVisualizer(KMeans(), k=(4, 8))

        visualizer.fit(docs)
        visualizer.poof()

        self.assert_images_similar(visualizer)

    def test_invalid_k(self):
        """
        Assert that invalid values of K raise exceptions
        """

        with pytest.raises(YellowbrickValueError):
            KElbowVisualizer(KMeans(), k=(1,2,3,4,5))

        with pytest.raises(YellowbrickValueError):
            KElbowVisualizer(KMeans(), k="foo")

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_distortion_metric(self):
        """
        Test the distortion metric of the k-elbow visualizer
        """
        visualizer = KElbowVisualizer(
            KMeans(random_state=0), k=5, metric="distortion", timings=False
        )
        visualizer.fit(X)

        expected = np.array([ 7.677785,  8.364319,  8.893634,  8.013021])
        assert len(visualizer.k_scores_) == 4

        visualizer.poof()
        self.assert_images_similar(visualizer)
        assert_array_almost_equal(visualizer.k_scores_, expected)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_silhouette_metric(self):
        """
        Test the silhouette metric of the k-elbow visualizer
        """
        visualizer = KElbowVisualizer(
            KMeans(random_state=0), k=5, metric="silhouette", timings=False
        )
        visualizer.fit(X)

        expected = np.array([ 0.691636,  0.456646,  0.255174,  0.239842])
        assert len(visualizer.k_scores_) == 4

        visualizer.poof()
        self.assert_images_similar(visualizer)
        assert_array_almost_equal(visualizer.k_scores_, expected)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_calinski_harabaz_metric(self):
        """
        Test the calinski-harabaz metric of the k-elbow visualizer
        """
        visualizer = KElbowVisualizer(
            KMeans(random_state=0), k=5,
            metric="calinski_harabaz", timings=False
        )
        visualizer.fit(X)
        assert len(visualizer.k_scores_) == 4

        expected = np.array([
            81.662726256035683, 50.992378259195554,
            40.952179227847012, 35.939494
        ])


        visualizer.poof()
        self.assert_images_similar(visualizer)
        assert_array_almost_equal(visualizer.k_scores_, expected)

    def test_bad_metric(self):
        """
        Assert KElbow raises an exception when a bad metric is supplied
        """
        with pytest.raises(YellowbrickValueError):
            KElbowVisualizer(KMeans(), k=5, metric="foo")

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    def test_timings(self):
        """
        Test the twinx double axes with k-elbow timings
        """
        visualizer = KElbowVisualizer(
            KMeans(random_state=0), k=5, timings=True
        )
        visualizer.fit(X)

        # Check that we kept track of time
        assert len(visualizer.k_timers_) == 4
        assert all([t > 0 for t in visualizer.k_timers_])

        # Check that we plotted time on a twinx
        assert hasattr(visualizer, "axes")
        assert len(visualizer.axes) == 2

        # delete the timings axes and
        # overwrite k_timers_, k_values_ for image similarity Tests
        visualizer.axes[1].remove()
        visualizer.k_timers_ = [
            0.01084589958190918, 0.011144161224365234,
            0.017028093338012695, 0.010634183883666992
        ]
        visualizer.k_values_ = [2, 3, 4, 5]

        # call draw again which is normally called in fit
        visualizer.draw()
        visualizer.poof()

        self.assert_images_similar(visualizer)
