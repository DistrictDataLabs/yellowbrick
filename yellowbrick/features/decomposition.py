##########################################################################
## Imports
##########################################################################

from .base import FeatureVisualizer

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

    def __init__(self, ax=None, n_components=None, scale=True, center=True, colormap='RdBu_r', 
                 **kwargs):

        super(ExplainedVariance, self).__init__(ax=ax, **kwargs)

        self.colormap = colormap
        self.explained_variance_ = None
        self.n_components = n_components
        self.center = center
        self.scale = scale
        self.pipeline = Pipeline([('scale', StandardScaler(with_mean=self.center,
                                                                   with_std=self.scale)), 
                                                                  ('pca', PCA(n_components=self.n_components))])

    def fit(self, X, y=None):
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        self.draw(self.pipeline.steps[-1][1].explained_variance_)
        return self.pipeline.transform(X)

    def draw(self, ev):
        self.ax.plot(ev)

    def finalize(self, **kwargs):
        # Set the title
        self.set_title('Explained Variance Plot')

        # Set the axes labels
        self.ax.set_ylabel('Explained Variance')
        self.ax.set_xlabel('Number of Components')