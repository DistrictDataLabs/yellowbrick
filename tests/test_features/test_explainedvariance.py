#!/usr/bin/env python3

"""
Test the ExplainedVariance visualizer
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest
import numpy as np

from tests.base import VisualTestCase
from tests.dataset import DatasetMixin, Dataset
from sklearn.datasets import make_classification

from yellowbrick.features import ExplainedVariance
from yellowbrick.exceptions import YellowbrickValueError

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
## ExplainedVariance Tests
##########################################################################

@pytest.mark.usefixtures('dataset')
class TestExplainedVariance(VisualTestCase, DatasetMixin):
    """
    Test the ExplainedVariance visualizer
    """

    def test_explainedvariance(self):
        """
        Assert image similarity on test dataset
        """
        visualizer = ExplainedVariance()
        visualizer.fit_transform(self.dataset.X)
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=(5 if sys.platform == 'win32' else 0.25))

    def test_explainedvariance_kaiser(self):
        """
        Assert image similarity with Kaiser threshold
        """
        visualizer = ExplainedVariance(kaiser=True)
        visualizer.fit_transform(self.dataset.X)
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=(5 if sys.platform == 'win32' else 0.25))

    def test_explainedvariance_kaiser_thresh(self):
        """
        Assert image similarity with Kaiser threshold
        that is specified by the user.
        """
        visualizer = ExplainedVariance(kaiser=True, kaiser_thresh=2.0)
        visualizer.fit_transform(self.dataset.X)
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=(5 if sys.platform == 'win32' else 0.25))
        
    def test_explainedvariance_scree(self):
        """
        Assert image similarity on Scree plot
        """
        visualizer = ExplainedVariance(scree=True)
        visualizer.fit_transform(self.dataset.X)
        visualizer.poof()
        self.assert_images_similar(visualizer, tol=(5 if sys.platform == 'win32' else 0.25))

    def test_explainedvariance_scree_kaiser_error(self):
        """
        Test that YellowbrickValueError is raised when
        kaiser and scree are both set to True.
        """

        with pytest.raises(YellowbrickValueError, match="Cannot plot a Kaiser threshold on a cumulative Scree plot"):
            visualizer = ExplainedVariance(scree=True, kaiser=True)
            visualizer.fit_transform(self.dataset.X)
            visualizer.poof()

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    @pytest.mark.skipif(pandas is None, reason="test requires Pandas")
    def test_integrated_explainedvariance_with_pandas(self):
        """
        Test ExplainedVariance with Pandas on the concrete dataset
        """
        # Load the data from the fixture
        concrete = self.load_pandas("concrete")
        features = concrete.columns.values
        
        X = concrete[features]

        # Test the visualizer
        visualizer = ExplainedVariance()
        visualizer.fit_transform_poof(X)
        self.assert_images_similar(visualizer, tol=0.25)

    @pytest.mark.xfail(
        sys.platform == 'win32', reason="images not close on windows"
    )
    @pytest.mark.skipif(pandas is None, reason="test requires Pandas")
    def test_integrated_explainedvariance_scree_with_pandas(self):
        """
        Test ExplainedVariance generating a Scree on a Pandas DataFrame
        """

        # Load the data from the fixture
        concrete = self.load_pandas("concrete")
        features = concrete.columns.values

        X = concrete[features]

        # Test the visualizer
        visualizer = ExplainedVariance(scree=True)
        visualizer.fit_transform_poof(X)
        self.assert_images_similar(visualizer, tol=0.25)
