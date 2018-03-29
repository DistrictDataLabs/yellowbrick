# yellowbrick.cluster.silhouette
# Implements visualizers using the silhouette metric for cluster evaluation.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Mon Mar 27 10:09:24 2017 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: silhouette.py [57b563b] benjamin@bengfort.com $

"""
Implements visualizers that use the silhouette metric for cluster evaluation.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from ..style import color_palette
from .base import ClusteringScoreVisualizer

from sklearn.metrics import silhouette_score, silhouette_samples


## Packages for export
__all__ = [
    "SilhouetteVisualizer"
]


##########################################################################
## Silhouette Method for K Selection
##########################################################################

class SilhouetteVisualizer(ClusteringScoreVisualizer):
    """
    TODO: Document this class!
    """

    def __init__(self, model, ax=None, **kwargs):
        super(SilhouetteVisualizer, self).__init__(model, ax=ax, **kwargs)

        # Visual Properties
        # TODO: Fix the color handling
        self.colormap = kwargs.get('colormap', 'set1')
        self.color = kwargs.get('color', None)

        # Required internal properties
        self.silhouette_score_ = None
        self.silhouette_samples_ = None
        self.n_samples = None
        self.n_clusters = None

    def fit(self, X, y=None, **kwargs):
        """
        Fits the model and generates the the silhouette visualization.

        TODO: decide to use this method or the score method to draw.
        NOTE: Probably this would be better in score, but the standard score
        is a little different and I'm not sure how it's used.
        """
        # Fit the wrapped estimator
        self.estimator.fit(X, y, **kwargs)

        # Get the properties of the dataset
        self.n_samples = X.shape[0]
        self.n_clusters = self.estimator.n_clusters

        # Compute the scores of the cluster
        labels = self.estimator.predict(X)
        self.silhouette_score_ = silhouette_score(X, labels)
        self.silhouette_samples_ = silhouette_samples(X, labels)

        # Draw the silhouette figure
        self.draw(labels)

        # Return the estimator
        return self

    def draw(self, labels):
        """
        Draw the silhouettes for each sample and the average score.

        Parameters
        ----------

        labels : array-like
            An array with the cluster label for each silhouette sample,
            usually computed with ``predict()``. Labels are not stored on the
            visualizer so that the figure can be redrawn with new data.
        """

        # Track the positions of the lines being drawn
        y_lower = 10 # The bottom of the silhouette

        # Get the colors from the various properties
        # TODO: Use resolve_colors instead of this
        colors = color_palette(self.colormap, self.n_clusters)

        # For each cluster, plot the silhouette scores
        for idx in range(self.n_clusters):

            # Collect silhouette scores for samples in the current cluster .
            values = self.silhouette_samples_[labels == idx]
            values.sort()

            # Compute the size of the cluster and find upper limit
            size = values.shape[0]
            y_upper = y_lower + size

            color = colors[idx]
            self.ax.fill_betweenx(
                np.arange(y_lower, y_upper), 0, values,
                facecolor=color, edgecolor=color, alpha=0.5
            )

            # Label the silhouette plots with their cluster numbers
            self.ax.text(-0.05, y_lower + 0.5 * size, str(idx))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10

        # The vertical line for average silhouette score of all the values
        self.ax.axvline(
            x=self.silhouette_score_, color="red", linestyle="--"
        )

        return self.ax

    def finalize(self):
        """
        Prepare the figure for rendering by setting the title and adjusting
        the limits on the axes, adding labels and a legend.
        """

        # Set the title
        self.set_title((
            "Silhouette Plot of {} Clustering for {} Samples in {} Centers"
        ).format(
            self.name, self.n_samples, self.n_clusters
        ))

        # Set the X and Y limits
        # The silhouette coefficient can range from -1, 1
        self.ax.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between
        # silhouette plots of individual clusters, to demarcate them clearly.
        self.ax.set_ylim([0, self.n_samples + (self.n_clusters + 1) * 10])

        # Set the x and y labels
        self.ax.set_xlabel("silhouette coefficient values")
        self.ax.set_ylabel("cluster label")

        # Set the ticks on the axis object.
        self.ax.set_yticks([])  # Clear the yaxis labels / ticks
        self.ax.set_xticks(np.linspace(-1,1,11))
