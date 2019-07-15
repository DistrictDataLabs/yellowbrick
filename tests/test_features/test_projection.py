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
from sklearn.manifold import LocallyLinearEmbedding

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

@pytest.fixture(scope="function")
def transformer(request):
    """
    Creates a PCA transformer. Acccepts the projection dimnesion as param and
    returns transformer accordingly. If parame is not specified then None is returned.
    """
    if (not hasattr(request, "param")):
        return
    pca_transformer = Pipeline([("scale", StandardScaler()),
                                    ("pca", PCA(request.param, random_state=2019))])
    # Set a class attribute for transformer
    request.cls.transformer = pca_transformer


@pytest.mark.usefixtures("discrete", "continuous", "transformer")
class TestProjectionVisualizerBase(VisualTestCase):
    
    @pytest.mark.parametrize("transformer", [2], indirect=True)
    def test_discrete_plot(self):

        X, y = self.discrete
        classes = ["a", "b", "c", "d", "e"]
        visualizer = ProjectionVisualizer(projection=2,
                                          colormap="plasma", classes=classes)
        visualizer.transformer=self.transformer
        transform_array = visualizer.fit_transform(X, y)
        assert(visualizer.classes_ == classes)
        visualizer.finalize()
        self.assert_images_similar(visualizer)
        assert transform_array.shape == (self.discrete.X.shape[0], 2)
        
    def test_continuous_plot(self):

        X, y = self.continuous
        manifold = Pipeline([
            ("pca", PCA(n_components=15, random_state=1998)),
            ("lle", LocallyLinearEmbedding(n_components=2, random_state=2019)),
        ])

        visualizer = ProjectionVisualizer(projection="2d")
        visualizer.transformer = manifold
        visualizer.fit_transform(X, y)
        visualizer.finalize()
        visualizer.cax.set_yticklabels([])
        self.assert_images_similar(visualizer)
        
    @pytest.mark.parametrize("transformer", [2], indirect=True)
    def test_continuous_when_target_discrete(self):
        """
        Test if data is dicrete but we override the target type to be continuous.
        """
        _, ax = plt.subplots()
        X, y = self.discrete
        visualizer = ProjectionVisualizer(ax=ax, projection="2D", 
                                          target_type="continuous", colormap="cool")
        visualizer.transformer = self.transformer
        visualizer.fit(X, y)
        visualizer.transform(X, y)
        visualizer.finalize()
        visualizer.cax.set_yticklabels([])
        self.assert_images_similar(visualizer)
        
    @pytest.mark.parametrize("transformer", [2], indirect=True)
    def test_single_plot(self):
        X, y = self.discrete
        visualizer = ProjectionVisualizer(projection=2,
                                          colormap="plasma")
        visualizer.transformer=self.transformer
        visualizer.fit_transform(X)
        visualizer.finalize()
        self.assert_images_similar(visualizer)

    
    @pytest.mark.parametrize("transformer", [3], indirect=True)
    def test_discrete_3d(self):
        X, y = self.discrete

        classes = ["a", "b", "c", "d", "e"]
        color = ["r", "b", "g", "m","c"]
        visualizer = ProjectionVisualizer(projection=3,
                                          color=color, classes=classes)
        visualizer.transformer=self.transformer
        visualizer.fit_transform(X, y)
        assert visualizer.classes_ == classes
        visualizer.finalize()
        self.assert_images_similar(visualizer)
    
    def test_3d_continuous_plot(self):
        X, y = self.continuous
        manifold = Pipeline([
            ("pca", PCA(n_components=15, random_state=1998)),
            ("lle", LocallyLinearEmbedding(n_components=3, random_state=2019)),
        ])

        visualizer = ProjectionVisualizer(projection="3D")
        visualizer.transformer = manifold
        visualizer.fit_transform(X, y)
        visualizer.finalize()
        visualizer.cbar.set_ticks([])
        self.assert_images_similar(visualizer)

    @pytest.mark.parametrize("transformer", [2], indirect=True)
    @mock.patch("yellowbrick.features.pca.plt.sca", autospec=True)
    def test_alpha_param(self, mock_sca):
        """
        Test that the user can supply an alpha param on instantiation
        """
        # Instantiate a prediction error plot, provide custom alpha
        X, y = self.discrete
        params = {"alpha": 0.3, "projection": 2}
        visualizer = ProjectionVisualizer(**params)
        visualizer.transformer=self.transformer
        visualizer.fit(X, y)
        visualizer.transform(X, y)
        assert visualizer.alpha == 0.3

        visualizer.ax = mock.MagicMock()
        visualizer.fit(X, y)
        visualizer.transform(X, y)

        # Test that alpha was passed to internal matplotlib scatterplot
        _, scatter_kwargs = visualizer.ax.scatter.call_args
        assert "alpha" in scatter_kwargs
        assert scatter_kwargs["alpha"] == 0.3

    # Check Errors
    @pytest.mark.parametrize("projection", ["4D", 1, "100d"])
    def test_wrong_projection_dimensions(self, projection):
        msg = "Projection dimensions must be either 2 or 3"
        with pytest.raises(YellowbrickValueError, match=msg):
            ProjectionVisualizer(projection=projection)

    
    def test_transform_without_fit(self):
        X, y = self.discrete
        visualizer = ProjectionVisualizer(projection="3D")
        msg = "try using fit_transform instead."
        with pytest.raises(AttributeError, match = msg):
            visualizer.transform(X, y)
    
    def test_target_not_label_encoded(self):
        X, y = self.discrete
        # Multiply every element by 10 to make non-label encoded
        y = y*10
        visualizer = ProjectionVisualizer()
        visualizer.transformer=self.transformer
        msg = "Target needs to be label encoded."
        with pytest.raises(YellowbrickValueError, match = msg):
            visualizer.fit_transform(X, y)        
        
    def test_y_required_for_discrete_and_continuous(self):
        X, y = self.discrete
        visualizer = ProjectionVisualizer()
        visualizer.transformer=self.transformer
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
