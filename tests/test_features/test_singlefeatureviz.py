# tests.test_features.test_singlefeatureviz
# Test the SingleFeatureViz visualizer
#
# Author:   Liam Schumm <lschumm@protonmail.com>
# Created:  Mon May 06 12:11:00 2019 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#

"""
Test the SingleFeatureViz visualizer
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin, Dataset
from sklearn.datasets import make_classification

from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.features.singlefeatureviz import *

try:
    import pandas
except ImportError:
    pandas = None


##########################################################################
## Fixtures
##########################################################################

@pytest.fixture(scope='class')
def dataset(request):
    """
    Creates a random multiclass classification dataset fixture
    """
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=451, flip_y=0,
        class_sep=3, scale=np.array([1.0, 2.0, 100.0, 20.0, 1.0])
    )

    dataset = Dataset(X, y)
    request.cls.dataset = dataset


##########################################################################
## SingleFeatureViz Tests
##########################################################################

@pytest.mark.usefixtures('dataset')
class TestSingleFeatureViz(VisualTestCase, DatasetMixin):
    """
    Test the SingleFeatureViz visualizer
    """

    def test_singlefeatureviz_defaultplot(self):
        """
        Assert image similarity on test dataset
        """

        visualizer = SingleFeatureViz(idx=1)
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_singlefeatureviz_stridx_defaultplot(self):
        """ 
        Assert image similarity on test dataset       
        """

        visualizer = SingleFeatureViz(idx="a", features=["a", "b", "c", "d", "e"])
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_singlefeatureviz_histplot(self):
        """
        Assert image similarity on test dataset
        """

        visualizer = SingleFeatureViz(idx=1)
        visualizer.fit_transform(self.dataset.X, self.dataset.y, plot_type="hist")
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_singlefeatureviz_stridx_histplot(self):
        """ 
        Assert image similarity on test dataset       
        """

        visualizer = SingleFeatureViz(idx="a", features=["a", "b", "c", "d", "e"], plot_type="hist")
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_singlefeatureviz_boxplot(self):
        """
        Assert image similarity on test dataset
        """

        visualizer = SingleFeatureViz(idx=1, plot_type="box")
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_singlefeatureviz_stridx_boxplot(self):
        """ 
        Assert image similarity on test dataset       
        """

        visualizer = SingleFeatureViz(idx="a", features=["a", "b", "c", "d", "e"], plot_type="box")
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_singlefeatureviz_violinplot(self):
        """
        Assert image similarity on test dataset
        """

        visualizer = SingleFeatureViz(idx=1, plot_type="violin")
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=0.25)

    def test_singlefeatureviz_stridx_no_features(self):
        with pytest.raises(YellowbrickValueError, match="A string index is specified without a features list on a NumPy array"):
            visualizer = SingleFeatureViz(idx="foo", plot_type="violin")
            visualizer.fit_transform(self.dataset.X, self.dataset.y)
            visualizer.poof()

    def test_singlefeatureviz_intidx_pandas(self):
        with pytest.raises(YellowbrickValueError, match="A string index is required for a Pandas DataFrame"):
            visualizer = SingleFeatureViz(idx="foo", plot_type="violin")
            visualizer.fit_transform(self.dataset.X, self.dataset.y)
            visualizer.poof()

    def test_singlefeatureviz_invalidplottype(self):
        with pytest.raises(YellowbrickValueError, match="my_fake_plot_type is not a valid plot_type for SingleFeatureViz"):
            visualizer = SingleFeatureViz(idx="foo", plot_type="my_fake_plot_type")
            visualizer.fit_transform(self.dataset.X, self.dataset.y)
            visualizer.poof()
        
    def test_singlefeatureviz_stridx_violinplot(self):
        """ 
        Assert image similarity on test dataset       
        """

        visualizer = SingleFeatureViz(idx="a", features=["a", "b", "c", "d", "e"], plot_type="violin")
        visualizer.fit_transform(self.dataset.X, self.dataset.y)
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=0.25)        

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    @pytest.mark.skipif(pandas is None, reason="test requires Pandas")
    def test_singlefeatureviz_with_pandas_intidx(self):
        """
        Test SingleFeatureViz with Pandas on the occupancy dataset
        """
        occupancy = self.load_pandas("occupancy")

        # Load the data from the fixture
        X = occupancy[[
            "temperature", "relative humidity", "light", "C02", "humidity"
        ]]
        y = occupancy['occupancy'].astype(int)


        with pytest.raises(YellowbrickValueError, match="A string index is required for a Pandas DataFrame"):
            visualizer = SingleFeatureViz(idx=0, plot_type="violin")
            visualizer.fit_transform(X, y)

    @pytest.mark.skipif(pandas is None, reason="test requires Pandas")
    def test_singlefeatureviz_with_pandas_stridx(self):
        """
        Test SingleFeatureViz with Pandas on the occupancy dataset
        """
        occupancy = self.load_pandas("occupancy")

        # Load the data from the fixture
        X = occupancy[[
            "temperature", "relative humidity", "light", "C02", "humidity"
        ]]
        y = occupancy['occupancy'].astype(int)

        # Test the visualizer
        visualizer = SingleFeatureViz(idx="relative humidity")
        visualizer.fit_transform_poof(X, y)
        self.assert_images_similar(visualizer)
