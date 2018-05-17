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
import numpy as np
import matplotlib.pyplot as plt

from six import string_types
from matplotlib import patches

from yellowbrick.utils.types import is_estimator
from yellowbrick.style import palettes, resolve_colors
from yellowbrick.features.base import FeatureVisualizer
from yellowbrick.exceptions import YellowbrickValueError, NotFitted

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

# Target type constants
AUTO = "auto"
SINGLE = "single"
DISCRETE = "discrete"
CONTINUOUS = "continuous"


##########################################################################
## Manifold Embeddings
##########################################################################

class Manifold(FeatureVisualizer):
    """
    The Manifold visualizer provides high dimensional visualization for feature
    analysis by embedding data into 2 dimensions using the sklearn.manifold
    package for manifold learning. In brief, manifold learning algorithms are
    unsuperivsed approaches to non-linear dimensionality reduction (unlike PCA
    or SVD) that help visualize latent structures in data.


    Notes
    -----
    ..see-also:: http://scikit-learn.org/stable/modules/manifold.html
    """

    ALGORITHMS = MANIFOLD_ALGORITHMS

    def __init__(
        self,
        ax=None,
        manifold="lle",
        n_neighbors=10,
        colors=None,
        target=AUTO,
        alpha=0.7,
        random_state=None,
        **kwargs
    ):
        super(Manifold, self).__init__(ax, **kwargs)
        self._name = None
        self._manifold = None
        self._target_color_type = None

        self.n_neighbors = n_neighbors
        self.colors = colors
        self.target = target
        self.alpha = alpha
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
        # Determine target type
        self._determine_target_color_type(y)

        # Compute classes and colors if target type is discrete
        if self._target_color_type == DISCRETE:
            self.classes_ = np.unique(y)

            color_kwargs = {'n_colors': len(self.classes_)}

            if isinstance(self.colors, string_types):
                color_kwargs['colormap'] = self.colors
            else:
                color_kwargs['colors'] = self.colors

            self._colors = resolve_colors(**color_kwargs)

        # Compute target range if colors are continuous
        elif self._target_color_type == CONTINUOUS:
            y = np.asarray(y)
            self.range_ = (y.min(), y.max())

        start = time.time()
        Xp = self.manifold.fit_transform(X)
        self.fit_time_ = time.time() - start

        self.draw(Xp, y)
        return self

    def transform(self, X):
        """
        Returns the transformed data points from the manifold embedding.
        """
        return self.manifold.transform(X)

    def draw(self, X, y=None):
        """
        Draws the points X on the axes.
        """
        scatter_kwargs = {"alpha": self.alpha}

        # Determine the colors
        if self._target_color_type == SINGLE:
            scatter_kwargs["c"] = "b"

        elif self._target_color_type == DISCRETE:
            if y is None:
                raise YellowbrickValueError("y is required for discrete target")

            scatter_kwargs["c"] = [
                self._colors[np.searchsorted(self.classes_, (yi))] for yi in y
            ]

        elif self._target_color_type == CONTINUOUS:
            if y is None:
                raise YellowbrickValueError("y is required for continuous target")

            # TODO manually make colorbar so we can draw it in finalize
            scatter_kwargs["c"] = y
            scatter_kwargs["cmap"] = self.colors or palettes.DEFAULT_SEQUENCE

        else:
            # Technically this should never be raised
            raise NotFitted("could not determine target color type")

        # Draw the scatter plot with the associated colors and alpha
        self._scatter = self.ax.scatter(X[:,0], X[:,1], **scatter_kwargs)

    def finalize(self):
        """
        Add title and modify axes to make the image ready for display
        """
        self.set_title(
            '{} Manifold (fit in {:0.2f} seconds)'.format(
                self._name, self.fit_time_
            )
        )
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        if self._target_color_type == DISCRETE:
            # Add the legend
            handles = [
                patches.Patch(color=self._colors[idx], label=self.classes_[idx])
                for idx in range(len(self.classes_))
            ]
            self.ax.legend(handles=handles)

        elif self._target_color_type == CONTINUOUS:
            # Add the color bar
            plt.colorbar(self._scatter, ax=self.ax)

    def _determine_target_color_type(self, y):
        """
        Determines the target color type from the vector y as follows:

            - if y is None: only a single color is used
            - if target is auto: determine if y is continuous or discrete
            - otherwise specify supplied target type

        This property will be used to compute the colors for each point.
        """
        if y is None:
            self._target_color_type = SINGLE
        elif self.target == "auto":
            # NOTE: See #73 for a generalization to use when implemented
            if len(np.unique(y)) < 10:
                self._target_color_type = DISCRETE
            else:
                self._target_color_type = CONTINUOUS
        else:
            self._target_color_type = self.target

        if self._target_color_type not in {SINGLE, DISCRETE, CONTINUOUS}:
            raise YellowbrickValueError((
                "could not determine target color type "
                "from target='{}' to '{}'"
            ).format(self.target, self._target_color_type))
