# yellowbrick.cluster.icdm
# Implements Intercluster Distance Map visualizations.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Tue Aug 21 11:56:53 2018 -0400
#
# ID: icdm.py [] benjamin@bengfort.com $

"""
Implements Intercluster Distance Map visualizations.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import MDS, TSNE

from .base import ClusteringScoreVisualizer

from ..utils.helpers import prop_to_size
from ..exceptions import YellowbrickValueError

from matplotlib.patches import Circle

##########################################################################
## InterclusterDistance Visualizer
##########################################################################

class InterclusterDistance(ClusteringScoreVisualizer):
    """
    Intercluster distance maps display an embedding of the cluster centers in
    2 dimensions with the distance to other centers preserved. E.g. the closer
    to centers are in the visualization, the closer they are in the original
    feature space. The clusters are sized according to a scoring metric. By
    default, they are sized by population, e.g. the number of instances that
    belong to each center. This gives a sense of the relative importance of
    clusters. Note however, that because two clusters overlap in the 2D space,
    it does not imply that they overlap in the original feature space.

    Parameters
    ----------

    Attributes
    ----------
    embedded_centers_ : array of shape (n_clusters, 2)
        The positions of all the cluster centers on the graph.

    scores_ : array of shape (n_clusters,)
        The scores of each cluster that determine its size on the graph.
    """

    EMBEDDINGS = {
        'mds': MDS,
        'tsne': TSNE,
    }

    SCORING = {
        'membership': np.bincount,
    }

    def __init__(self, model, ax=None, min_size=400, max_size=25000, legend=True,
                 embedding='mds', scoring='membership', **kwargs):
        # Initialize the visualizer bases
        super(InterclusterDistance, self).__init__(model, ax=ax, **kwargs)

        # Ensure that a valid embedding is passed.
        if embedding.lower() not in self.EMBEDDINGS:
            raise YellowbrickValueError(
                "unknown embedding '{}', chose from '{}'".format(
                    embedding, ", ".join(self.EMBEDDINGS.keys())
                )
            )

        # Ensure that a valid scoring metric is passed.
        if scoring.lower() not in self.SCORING:
            raise YellowbrickValueError(
                "unknown scoring '{}', chose from '{}'".format(
                    scoring, ", ".join(self.SCORING.keys())
                )
            )

        # Set decomposition properties
        self.embedding = embedding
        self.scoring = scoring

        # Set visual properties
        self.min_size = min_size
        self.max_size = max_size
        self.legend = legend

        # Colors are currently hardcoded, need to compute face and edge color
        # from this color based on the alpha of the cluster center.
        self.facecolor = "#2e719344"
        self.edgecolor = "#2e719399"

    def fit(self, X, y=None):
        """
        Fit the clustering model, computing the centers then embeds the centers
        into 2D space using the embedding method specified.
        """
        # Fit the underlying estimator
        self.estimator.fit(X, y)

        # Get the centers
        # TODO: is this how sklearn stores all centers in the model?
        C = self.estimator.cluster_centers_

        # Embed the centers
        transformer = self.EMBEDDINGS[self.embedding.lower()](n_components=2)
        self.embedded_centers_ = transformer.fit_transform(C)

        # Compute the score of the centers
        self.scores_ = self.SCORING[self.scoring.lower()](self.estimator.predict(X))

        # Draw the clusters
        self.draw()

        # Fit returns self
        return self

    def draw(self):
        """
        Draw the embedded centers with their sizes on the visualization.
        """
        # Compute the sizes of the markers from their score
        sizes = self._scores_to_size(self.scores_)

        # Draw the scatter plots with associated sizes on the graph
        self.ax.scatter(
            self.embedded_centers_[:,0], self.embedded_centers_[:,1],
            s=sizes, c=self.facecolor, edgecolor=self.edgecolor, linewidth=1,
        )

        # Annotate the clusters with their labels
        # TODO: font size is hardcoded here, how to handle?
        for i, pt in enumerate(self.embedded_centers_):
            self.ax.text(
                s=str(i), x=pt[0], y=pt[1], va="center", ha="center",
                fontweight="bold", size=13
            )

    def finalize(self):
        """
        Finalize the visualization to create an "origin grid" feel instead of
        the default matplotlib feel. Set the title, remove spines, and label
        the grid with components. This function also adds a legend from the
        sizes if required.
        """
        self.set_title("{} Intercluster Distance Map (via {})".format(
            self.estimator.__class__.__name__, self.embedding.upper()
        ))

        # Create the origin grid
        self.ax.set_xticks([0])
        self.ax.set_yticks([0])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_xlabel("PC2")
        self.ax.set_ylabel("PC1")

        # Make the legend by pushing over the current axes
        # TODO: fix and generalize this similar to ResidualsPlot.hax
        if self.legend:
            lax = plt.gcf().add_axes([.9, 0.25, 0.3333, 0.5], frameon=False, facecolor="none")
            self._make_size_legend(self.scores_, lax)

    def _scores_to_size(self, scores):
        """
        Returns the marker size (in points, e.g. area of the circle) based on
        the scores, using the prop_to_size scaling mechanism.
        """
        # NOTE: log and power are hardcoded, should we allow the user to specify?
        return prop_to_size(
            scores, mi=self.min_size, ma=self.max_size, log=False, power=0.5
        )

    def _make_size_legend(self, scores, ax):
        """
        Draw a legend that shows relative sizes of the clusters at the 25th,
        50th, and 75th percentile based on the current scoring metric.
        """
        # Compute the size of the markers and scale them to our figure size
        areas = self._scores_to_size(scores)
        radii = np.sqrt(areas / np.pi)
        scaled = np.interp(radii, (radii.min(), radii.max()), (.1, 1))

        # Compute the locations of the 25th, 50th, and 75th percentiles of the score
        indices = np.array([
            np.where(scores==np.percentile(scores, p, interpolation='nearest'))[0][0]
            for p in (25, 50, 75)
        ])

        # Draw circles with their various sizes as the legend
        for idx in indices:
            center = (-0.30, 1-scaled[idx])
            c = Circle(center, scaled[idx], facecolor="none", edgecolor="#2e7193", linewidth=1.5, linestyle="--", label="bob")
            ax.add_patch(c)

            ax.annotate(
                scores[idx], (-0.30, 1-(2*scaled[idx])), xytext=(1, 1-(2*scaled[idx])),
                arrowprops=dict(arrowstyle="wedge", color="#2e7193"), va='center', ha='center',
            )

        # Draw size legend title
        ax.text(s="membership", x=0, y=1.2, va='center', ha='center')

        ax.set_xlim(-1.4,1.4)
        ax.set_ylim(-1.4,1.4)
        ax.set_xticks([])
        ax.set_yticks([])
        for name in ax.spines:
            ax.spines[name].set_visible(False)

        ax.grid(False)
