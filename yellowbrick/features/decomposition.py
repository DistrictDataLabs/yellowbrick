##########################################################################
## Imports
##########################################################################

from .base import FeatureVisualizer
from yellowbrick.style import palettes

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

##########################################################################
## Decomposition Feature Visualizer
##########################################################################

class ExplainedVariance(FeatureVisualizer):
    """

    Parameters
    ----------
    
    

    Examples
    --------

    >>> visualizer = ExplainedVariance()
    >>> visualizer.fit(X)
    >>> visualizer.transform(X)
    >>> visualizer.poof()

    Notes
    -----
    
    """

    def __init__(self, ax=None, n_components=None, scale=True, center=True, colormap=palettes.DEFAULT_SEQUENCE, 
                 **kwargs):

        super(ExplainedVariance, self).__init__(ax=ax, **kwargs)

        self.colormap = colormap
        self.n_components = n_components
        self.center = center
        self.scale = scale
        self.pipeline = Pipeline([('scale', StandardScaler(with_mean=self.center,
                                                                   with_std=self.scale)), 
                                                                  ('pca', PCA(n_components=self.n_components))])
        self.pca_features = None

    @property
    def explained_variance_(self):
        return self.pipeline.steps[-1][1].explained_variance_

    def fit(self, X, y=None):
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        self.pca_features = self.pipeline.transform(X)
        self.draw()
        return self.pca_features

    def draw(self):
        X = self.explained_variance_
        self.ax.plot(X)
        return self.ax

    def finalize(self, **kwargs):
        # Set the title
        self.set_title('Explained Variance Plot')

        # Set the axes labels
        self.ax.set_ylabel('Explained Variance')
        self.ax.set_xlabel('Number of Components')
