# tests.test_features.test_feature_correlation
# Test the feature correlation visualizers
#
# Author:  Zijie (ZJ) Poh
# Created: Tue Jul 31 20:21:32 2018 -0700
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_feature_correlation.py [33aec16] 8103276+zjpoh@users.noreply.github.com $

"""
Test the feature correlation to dependent variable visualizer.
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest
import numpy as np
import numpy.testing as npt


from yellowbrick.target import FeatureCorrelation, feature_correlation
from yellowbrick.exceptions import YellowbrickValueError, YellowbrickWarning

from sklearn import datasets
from tests.base import VisualTestCase

try:
    import pandas as pd
except ImportError:
    pd = None


##########################################################################
## Feature Correlation Tests
##########################################################################


class TestFeatureCorrelationVisualizer(VisualTestCase):
    """
    FeatureCorrelation visualizer
    """

    data = datasets.load_diabetes()
    X, y = data["data"], data["target"]
    labels = data["feature_names"]

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_feature_correlation_integrated_pearson(self):
        """
        Test FeatureCorrelation visualizer with pearson correlation
        coefficient
        """
        viz = FeatureCorrelation()
        viz.fit(self.X, self.y)
        viz.finalize()

        self.assert_images_similar(viz)

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_feature_correlation_integrated_mutual_info_regression(self):
        """
        Test FeatureCorrelation visualizer with mutual information regression
        """
        viz = FeatureCorrelation(method="mutual_info-regression")
        viz.fit(self.X, self.y, random_state=23456)
        viz.finalize()

        self.assert_images_similar(viz)

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_feature_correlation_integrated_mutual_info_classification(self):
        """
        Test FeatureCorrelation visualizer with mutual information
        on wine dataset (classification)
        """
        data = datasets.load_wine()
        X, y = data["data"], data["target"]

        viz = FeatureCorrelation(method="mutual_info-classification")
        viz.fit(X, y, random_state=12345)
        viz.finalize()

        self.assert_images_similar(viz)

    def test_feature_correlation_method_not_implemented(self):
        """
        Test FeatureCorrelation visualizer with unknown method
        """
        method = "foo"
        e = "Method foo not implement; choose from *"
        with pytest.raises(YellowbrickValueError, match=e):
            FeatureCorrelation(method=method)

    def test_feature_correlation_labels_from_index(self):
        """
        Test getting feature labels from index
        """
        viz = FeatureCorrelation()
        viz.fit(self.X, self.y)

        npt.assert_array_equal(viz.features_, np.arange(self.X.shape[1]))

    def test_feature_correlation_labels(self):
        """
        Test labels as feature labels
        """
        viz = FeatureCorrelation(labels=self.labels)
        viz.fit(self.X, self.y)

        npt.assert_array_equal(viz.features_, self.labels)

    @pytest.mark.skipif(pd is None, reason="requires pandas")
    def test_feature_correlation_labels_from_dataframe(self):
        """
        Test getting feature labels from DataFrame
        """
        X_pd = pd.DataFrame(self.X, columns=self.labels)

        viz = FeatureCorrelation()
        viz.fit(X_pd, self.y)

        npt.assert_array_equal(viz.features_, self.labels)

    def test_feature_correlation_select_feature_by_index_out_of_range(self):
        """
        Test selecting feature by feature index but index is out of range
        """
        e = "Feature index is out of range"
        with pytest.raises(YellowbrickValueError, match=e):
            viz = FeatureCorrelation(feature_index=[0, 2, 10])
            viz.fit(self.X, self.y)

    def test_feature_correlation_select_feature_by_index(self):
        """
        Test selecting feature by index
        """
        viz = FeatureCorrelation(feature_index=[0, 2, 3])
        viz.fit(self.X, self.y)

        assert viz.scores_.shape[0] == 3

    def test_feature_correlation_select_feature_by_index_and_name(self):
        """
        Test selecting feature warning when both index and names are provided
        """
        feature_index = [0, 2, 3]
        feature_names = ["age"]

        e = (
            "Both feature_index and feature_names are specified. "
            "feature_names is ignored"
        )
        with pytest.raises(YellowbrickWarning, match=e):
            viz = FeatureCorrelation(
                feature_index=feature_index, feature_names=feature_names
            )
            viz.fit(self.X, self.y)
            assert viz.scores_.shape[0] == 3

    def test_feature_correlation_select_feature_by_name_no_labels(self):
        """
        Test selecting feature by feature names with labels is not supplied
        """
        feature_names = ["age"]

        e = "age not in labels"
        with pytest.raises(YellowbrickValueError, match=e):
            viz = FeatureCorrelation(feature_names=feature_names)
            viz.fit(self.X, self.y)

    def test_feature_correlation_select_feature_by_name(self):
        """
        Test selecting feature by feature names
        """
        feature_names = ["age", "sex", "bp", "s5"]

        viz = FeatureCorrelation(labels=self.labels, feature_names=feature_names)
        viz.fit(self.X, self.y)

        npt.assert_array_equal(viz.features_, feature_names)

    def test_feature_correlation_sort(self):
        """
        Test sorting of correlation
        """
        viz = FeatureCorrelation(sort=True)
        viz.fit(self.X, self.y)

        assert np.all(viz.scores_[:-1] <= viz.scores_[1:])

    @pytest.mark.xfail(sys.platform == "win32", reason="images not close on windows")
    def test_feature_correlation_quick_method(self):
        """
        Test sorting of correlation
        """
        g = feature_correlation.feature_correlation(self.X, self.y, labels=self.labels, show=False)

        self.assert_images_similar(g)
