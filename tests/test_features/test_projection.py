import pytest
import matplotlib.pyplot as plt

from yellowbrick.features.projection import ProjectionVisualizer
from yellowbrick.exceptions import YellowbrickValueError

from tests.base import VisualTestCase
from ..fixtures import Dataset
from unittest import mock

from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  StandardScaler
from sklearn.decomposition import PCA

##########################################################################
## Fixtures
##########################################################################

@pytest.fixture(scope="class")
def discrete(request):
    """
    Creare a random classification fixture.
    """
    X, y = make_classification(
        n_samples=400, n_features=12, n_informative=10, n_redundant=0,
        n_classes=5, random_state=2019)

    # Set a class attribute for discrete data
    request.cls.discrete = Dataset(X, y)

@pytest.fixture(scope="class")
def continuous(request):
    """
    Creates a random regressor fixture.
    """
    X, y = make_regression(
        n_samples=500, n_features=22, n_informative=8, random_state=2019
    )

    # Set a class attribute for continuous data
    request.cls.continuous = Dataset(X, y)

class MockVisualizer(ProjectionVisualizer):
    
    def __init__(self, ax=None, features=None, classes=None, color=None,
             colormap=None, target_type="auto", projection=2,
             alpha=0.75,**kwargs):
        
        super(MockVisualizer, self).__init__(ax=ax,
                                             features=features, classes=classes,
                                             color=color, colormap=colormap,
                                             target_type=target_type,
                                             projection=projection, alpha=alpha,
                                             **kwargs)
        
        self.pca_transformer = Pipeline([("scale", StandardScaler()),
                                    ("pca", PCA(self.projection, random_state=2019))])
    
    def fit(self, X, y=None):
        super(MockVisualizer, self).fit(X, y)
        self.pca_transformer.fit(X)
        return self
    
    def transform(self, X, y=None):
        try:
            Xp = self.pca_transformer.transform(X)
        except AttributeError as e:
            raise AttributeError(str(e) + " try using fit_transform instead.")
        self.draw(Xp, y)
        return Xp
        

@pytest.mark.usefixtures("discrete", "continuous")
class TestProjectionVisualizerBase(VisualTestCase):
    
    def test_discrete_plot(self):
        """
        Test the visualizer with discrete target.
        """
        X, y = self.discrete
        classes = ["a", "b", "c", "d", "e"]
        visualizer = MockVisualizer(projection=2, colormap="plasma", classes=classes)
        X_prime = visualizer.fit_transform(X, y)
        assert(visualizer.classes_ == classes)
        visualizer.finalize()
        self.assert_images_similar(visualizer)
        assert X_prime.shape == (self.discrete.X.shape[0], 2)
        
    def test_continuous_plot(self):
        """
        Tests the visualizer with continuous target.
        """
        X, y = self.continuous
        visualizer = MockVisualizer(projection="2d")
        visualizer.fit_transform(X, y)
        visualizer.finalize()
        visualizer.cax.set_yticklabels([])
        self.assert_images_similar(visualizer)

    def test_continuous_when_target_discrete(self):
        """
        Test if data is dicrete but we override the target type to be continuous.
        """
        _, ax = plt.subplots()
        X, y = self.discrete
        visualizer = MockVisualizer(ax=ax, projection="2D", 
                                          target_type="continuous", colormap="cool")
        visualizer.fit(X, y)
        visualizer.transform(X, y)
        visualizer.finalize()
        visualizer.cax.set_yticklabels([])
        self.assert_images_similar(visualizer)
        
    def test_single_plot(self):
        """
        Tests when y is not specified. In such cases Visualizer consider it to 
        be a single target type.
        """
        X, y = self.discrete
        visualizer = MockVisualizer(projection=2,
                                          colormap="plasma")
        visualizer.fit_transform(X)
        visualizer.finalize()
        self.assert_images_similar(visualizer)

    def test_discrete_3d(self):
        """
        Test visualizer for 3 dimensional discrete plots
        """
        X, y = self.discrete

        classes = ["a", "b", "c", "d", "e"]
        color = ["r", "b", "g", "m","c"]
        visualizer = MockVisualizer(projection=3,
                                          color=color, classes=classes)
        visualizer.fit_transform(X, y)
        assert visualizer.classes_ == classes
        visualizer.finalize()
        self.assert_images_similar(visualizer)
    
    def test_3d_continuous_plot(self):
        """
        Tests visualizer for 3 dimensional continuous plots
        """
        X, y = self.continuous
        visualizer = MockVisualizer(projection="3D")
        visualizer.fit_transform(X, y)
        visualizer.finalize()
        visualizer.cbar.set_ticks([])
        self.assert_images_similar(visualizer)

    def test_alpha_param(self):
        """
        Test that the user can supply an alpha param on instantiation
        """
        # Instantiate a prediction error plot, provide custom alpha
        X, y = self.discrete
        params = {"alpha": 0.3, "projection": 2}
        visualizer = MockVisualizer(**params)
        visualizer.ax = mock.MagicMock()
        visualizer.fit(X, y)
        visualizer.transform(X, y)

        assert visualizer.alpha == 0.3

        # Test that alpha was passed to internal matplotlib scatterplot
        _, scatter_kwargs = visualizer.ax.scatter.call_args
        assert "alpha" in scatter_kwargs
        assert scatter_kwargs["alpha"] == 0.3

    # Check Errors
    @pytest.mark.parametrize("projection", ["4D", 1, "100d"])
    def test_wrong_projection_dimensions(self, projection):
        """
        Ensure that error is raised when wrong projecting dimensions are passed.
        """
        msg = "Projection dimensions must be either 2 or 3"
        with pytest.raises(YellowbrickValueError, match=msg):
            MockVisualizer(projection=projection)
    
    def test_target_not_label_encoded(self):
        """
        Ensures that proper error is raised fir non label encoded classes.
        """
        X, y = self.discrete
        # Multiply every element by 10 to make non-label encoded
        y = y*10
        visualizer = MockVisualizer()
        msg = "Target needs to be label encoded."
        with pytest.raises(YellowbrickValueError, match = msg):
            visualizer.fit_transform(X, y)        
        
    def test_y_required_for_discrete_and_continuous(self):
        """
        Raises an error if target is specified in fit but not in transform.
        """
        X, y = self.discrete
        visualizer = MockVisualizer()
        visualizer.fit(X, y)
        msg = "y is required for discrete target"
        with pytest.raises(YellowbrickValueError, match = msg):
            visualizer.transform(X)
        
        # Continuous target
        X, y = self.continuous
        visualizer.fit(X, y)
        msg = "y is required for continuous target"
        with pytest.raises(YellowbrickValueError, match = msg):
            visualizer.transform(X)
