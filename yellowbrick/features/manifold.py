# yellowbrick.features.manifold
# Use manifold algorithms for high dimensional visualization.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Sat May 12 11:25:24 2018 -0400
#
# ID: manifold.py [] benjamin@bengfort.com $

"""
Use manifold algorithms for high dimensional visualization.
"""

##########################################################################
## Imports
##########################################################################

import time

from matplotlib.pyplot import cm

from yellowbrick.utils.types import is_estimator
from yellowbrick.features.base import FeatureVisualizer
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.base import clone
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap, MDS, TSNE, SpectralEmbedding


##########################################################################
## Supported manifold algorithms by name lookup
##########################################################################

MANIFOLD_ALGORITHMS = {
    "lle": LocallyLinearEmbedding(method="standard", eigen_solver='auto'),
    "ltsa":LocallyLinearEmbedding(method="ltsa", eigen_solver='auto'),
    "hessian": LocallyLinearEmbedding(method="hessian", eigen_solver='auto'),
    "modified": LocallyLinearEmbedding(method="modified", eigen_solver='auto'),
    "isomap": Isomap(),
    "mds": MDS(),
    "spectral": SpectralEmbedding(),
    "tsne": TSNE(init='pca'),
}

MANIFOLD_NAMES = {
    "lle": "Locally Linear Embedding",
    "ltsa": "LTSA LLE",
    "hessian": "Hessian LLE",
    "modified": "Modified LLE",
    "isomap": "Isomap",
    "mds": "MDS",
    "spectral": "Spectral Embedding",
    "tsne": "t-SNE",
}



##########################################################################
## 2D Manifold Projections
##########################################################################

class Manifold(FeatureVisualizer):
    """

    """

    ALGORITHMS = MANIFOLD_ALGORITHMS

    def __init__(
        self,
        ax=None,
        manifold="lle",
        n_neighbors=10,
        random_state=None,
        **kwargs
    ):
        super().__init__(ax, **kwargs)
        self._name = None
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.manifold = manifold # must be set last

    @property
    def manifold(self):
        return self._manifold

    @manifold.setter
    def manifold(self, transformer):
        """
        Creates the manifold estimator if a string value is passed in,
        validates other objects passed in.
        """
        if not is_estimator(transformer):
            if transformer not in self.ALGORITHMS:
                raise YellowbrickValueError(
                    "could not create manifold for '%s'".format(str(transformer))
                )

            # Create a new transformer with the specified params
            self._name = MANIFOLD_NAMES[transformer]
            transformer = clone(self.ALGORITHMS[transformer])
            params = {
                "n_components": 2,
                "n_neighbors": self.n_neighbors,
                "random_state": self.random_state,
            }

            for param in list(params.keys()):
                if param not in transformer.get_params():
                    del params[param]

            transformer.set_params(**params)

        self._manifold = transformer
        if self._name is None:
            self._name = self._manifold.__class__.__name__

    def fit(self, X, y=None):
        """
        Fits the manifold on X and transforms the data to plot it on the axes.
        The optional y specified can be used to declare discrete colors.
        """
        start = time.time()
        Xp = self.manifold.fit_transform(X)
        self.fit_time_ = time.time() - start
        self.draw(Xp, y)

    def transform(self, X):
        """
        Returns the transformed data points from the manifold embedding.
        """
        return self.manifold.transform(X)

    def draw(self, X, y):
        """
        Draws the points X on the axes.
        """
        self.ax.scatter(X[:,0], X[:,1], c=y, alpha=0.7, cmap=cm.Spectral)

    def finalize(self):
        """
        Add title and modify axes to make the image ready for display
        """
        self.set_title('{} Manifold (fit in {:0.2f} seconds)'.format(self._name, self.fit_time_))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
