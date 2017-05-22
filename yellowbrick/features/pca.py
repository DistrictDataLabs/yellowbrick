import matplotlib.pyplot as plt

from yellowbrick.features.base import DataVisualizer

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
##########################################################################
## Rank 2D PCA Visualizer
##########################################################################
class PCA2D(DataVisualizer):

    def __init__(self, ax=None, scale=True, center=False, col=None,
                 colormap='RdBu_r', **kwargs):
        super(PCA2D, self).__init__(ax=ax, **kwargs)
        # Data Parameters
        self.col = col
        self.pca_features_ = None
        self.scale = scale
        self.center = center
        self.pca_transformer = Pipeline([('scale', StandardScaler(with_mean=self.center, 
                                                                  with_std=self.scale)),
                                         ('pca', PCA(2))])
        # Visual Parameters
        self.colormap = colormap

    def fit(self, X, y=None, **kwargs):
        self.pca_transformer.fit(X)
        return self

    def transform(self, X, y=None, **kwargs):
        self.pca_features_ = self.pca_transformer.transform(X)
        self.draw()
        return self.pca_features_

    def draw(self, **kwargs):
        X = self.pca_features_
        if self.ax is None:
            self.ax = self.gca()

        self.ax.scatter(X[:, 0], X[:, 1], c=self.col, cmap=plt.cm.Paired)
        return self.ax

    def finalize(self, **kwargs):
        # Set the title
        self.set_title('Principal Component Plot')

        # Set the axes labels
        self.ax.set_ylabel('Principal Component 2')
        self.ax.set_xlabel('Principal Component 1')


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    params = {'scale': True, 'center': False, 'col': y}
    visualizer = PCA2D(**params)
    visualizer.fit(X)
    visualizer.transform(X)
    visualizer.poof()
