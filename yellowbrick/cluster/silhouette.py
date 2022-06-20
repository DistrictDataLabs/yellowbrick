# yellowbrick.cluster.silhouette
# Implements visualizers using the silhouette metric for cluster evaluation.
#
# Author:   Benjamin Bengfort
# Author:   Rebecca Bilbro
# Created:  Mon Mar 27 10:09:24 2017 -0400
#
# Copyright (C) 2017 The scikit-yb developers
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
import matplotlib.ticker as ticker

from sklearn.metrics import silhouette_score, silhouette_samples

from yellowbrick.utils import check_fitted
from yellowbrick.style import resolve_colors
from yellowbrick.cluster.base import ClusteringScoreVisualizer

## Packages for export
__all__ = ["SilhouetteVisualizer", "silhouette_visualizer"]


##########################################################################
## Silhouette Method for K Selection
##########################################################################


class SilhouetteVisualizer(ClusteringScoreVisualizer):
    """
    The Silhouette Visualizer displays the silhouette coefficient for each
    sample on a per-cluster basis, visually evaluating the density and
    separation between clusters. The score is calculated by averaging the
    silhouette coefficient for each sample, computed as the difference
    between the average intra-cluster distance and the mean nearest-cluster
    distance for each sample, normalized by the maximum value. This produces a
    score between -1 and +1, where scores near +1 indicate high separation
    and scores near -1 indicate that the samples may have been assigned to
    the wrong cluster.

    In SilhouetteVisualizer plots, clusters with higher scores have wider
    silhouettes, but clusters that are less cohesive will fall short of the
    average score across all clusters, which is plotted as a vertical dotted
    red line.

    This is particularly useful for determining cluster imbalance, or for
    selecting a value for K by comparing multiple visualizers.

    Parameters
    ----------
    estimator : a Scikit-Learn clusterer
        Should be an instance of a centroidal clustering algorithm (``KMeans``
        or ``MiniBatchKMeans``). If the estimator is not fitted, it is fit when
        the visualizer is fitted, unless otherwise specified by ``is_fitted``.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    colors : iterable or string, default: None
        A collection of colors to use for each cluster group. If there are
        fewer colors than cluster groups, colors will repeat. May also be a
        Yellowbrick or matplotlib colormap string.

    is_fitted : bool or str, default='auto'
        Specify if the wrapped estimator is already fitted. If False, the
        estimator will be fit when the visualizer is fit, otherwise, the
        estimator will not be modified. If 'auto' (default), a helper method
        will check if the estimator is fitted before fitting it again.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    silhouette_score_ : float
        Mean Silhouette Coefficient for all samples. Computed via scikit-learn
        `sklearn.metrics.silhouette_score`.

    silhouette_samples_ : array, shape = [n_samples]
        Silhouette Coefficient for each samples. Computed via scikit-learn
        `sklearn.metrics.silhouette_samples`.

    n_samples_ : integer
        Number of total samples in the dataset (X.shape[0])

    n_clusters_ : integer
        Number of clusters (e.g. n_clusters or k value) passed to internal
        scikit-learn model.

    y_tick_pos_ : array of shape (n_clusters,)
        The computed center positions of each cluster on the y-axis

    Examples
    --------

    >>> from yellowbrick.cluster import SilhouetteVisualizer
    >>> from sklearn.cluster import KMeans
    >>> model = SilhouetteVisualizer(KMeans(10))
    >>> model.fit(X)
    >>> model.show()
    """

    def __init__(self, estimator, ax=None, colors=None, is_fitted="auto", **kwargs):

        # Initialize the visualizer bases
        super(SilhouetteVisualizer, self).__init__(estimator, ax=ax, **kwargs)

        # Visual Properties
        # Use colors if it is given, otherwise attempt to use colormap which
        # which will override colors. If neither is found, default to None.
        # The colormap may yet still be found in resolve_colors
        self.colors = colors
        if "colormap" in kwargs:
            self.colors = kwargs["colormap"]

    def fit(self, X, y=None, **kwargs):
        """
        Fits the model and generates the silhouette visualization.
        """
        # TODO: decide to use this method or the score method to draw.
        # NOTE: Probably this would be better in score, but the standard score
        # is a little different and I'm not sure how it's used.

        if not check_fitted(self.estimator, is_fitted_by=self.is_fitted):
            # Fit the wrapped estimator
            self.estimator.fit(X, y, **kwargs)

        # Get the properties of the dataset
        self.n_samples_ = X.shape[0]
        self.n_clusters_ = self.estimator.n_clusters

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
        y_lower = 10  # The bottom of the silhouette

        # Get the colors from the various properties
        color_kwargs = {"n_colors": self.n_clusters_}

        if self.colors is None:
            color_kwargs["colormap"] = "Set1"
        elif isinstance(self.colors, str):
            color_kwargs["colormap"] = self.colors
        else:
            color_kwargs["colors"] = self.colors

        colors = resolve_colors(**color_kwargs)

        # For each cluster, plot the silhouette scores
        self.y_tick_pos_ = []
        for idx in range(self.n_clusters_):

            # Collect silhouette scores for samples in the current cluster .
            values = self.silhouette_samples_[labels == idx]
            values.sort()

            # Compute the size of the cluster and find upper limit
            size = values.shape[0]
            y_upper = y_lower + size

            color = colors[idx]
            self.ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                values,
                facecolor=color,
                edgecolor=color,
                alpha=0.5,
            )

            # Collect the tick position for each cluster
            self.y_tick_pos_.append(y_lower + 0.5 * size)

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10

        # The vertical line for average silhouette score of all the values
        self.ax.axvline(
            x=self.silhouette_score_,
            color="red",
            linestyle="--",
            label="Average Silhouette Score",
        )

        return self.ax

    def finalize(self):
        """
        Prepare the figure for rendering by setting the title and adjusting
        the limits on the axes, adding labels and a legend.
        """

        # Set the title
        self.set_title(
            ("Silhouette Plot of {} Clustering for {} Samples in {} Centers").format(
                self.name, self.n_samples_, self.n_clusters_
            )
        )

        # Set the X and Y limits
        # The silhouette coefficient can range from -1, 1;
        # but here we scale the plot according to our visualizations

        # l_xlim and u_xlim are lower and upper limits of the x-axis,
        # set according to our calculated max and min score with necessary padding
        l_xlim = max(-1, min(-0.1, round(min(self.silhouette_samples_) - 0.1, 1)))
        u_xlim = min(1, round(max(self.silhouette_samples_) + 0.1, 1))
        self.ax.set_xlim([l_xlim, u_xlim])

        # The (n_clusters_+1)*10 is for inserting blank space between
        # silhouette plots of individual clusters, to demarcate them clearly.
        self.ax.set_ylim([0, self.n_samples_ + (self.n_clusters_ + 1) * 10])

        # Set the x and y labels
        self.ax.set_xlabel("silhouette coefficient values")
        self.ax.set_ylabel("cluster label")

        # Set the ticks on the axis object.
        self.ax.set_yticks(self.y_tick_pos_)
        self.ax.set_yticklabels(str(idx) for idx in range(self.n_clusters_))
        # Set the ticks at multiples of 0.1
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

        # Show legend (Average Silhouette Score axis)
        self.ax.legend(loc="best")


##########################################################################
## Quick Method
##########################################################################


def silhouette_visualizer(
    estimator, X, y=None, ax=None, colors=None, is_fitted="auto", show=True, **kwargs
):
    """Quick Method:
    The Silhouette Visualizer displays the silhouette coefficient for each
    sample on a per-cluster basis, visually evaluating the density and
    separation between clusters. The score is calculated by averaging the
    silhouette coefficient for each sample, computed as the difference
    between the average intra-cluster distance and the mean nearest-cluster
    distance for each sample, normalized by the maximum value. This produces a
    score between -1 and +1, where scores near +1 indicate high separation
    and scores near -1 indicate that the samples may have been assigned to
    the wrong cluster.

    Parameters
    ----------
    estimator : a Scikit-Learn clusterer
        Should be an instance of a centroidal clustering algorithm (``KMeans``
        or ``MiniBatchKMeans``). If the estimator is not fitted, it is fit when
        the visualizer is fitted, unless otherwise specified by ``is_fitted``.

    X : array-like of shape (n, m)
        A matrix or data frame with n instances and m features

    y : array-like of shape (n,), optional
        A vector or series representing the target for each instance

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    colors : iterable or string, default: None
        A collection of colors to use for each cluster group. If there are
        fewer colors than cluster groups, colors will repeat. May also be a
        Yellowbrick or matplotlib colormap string.

    is_fitted : bool or str, default='auto'
        Specify if the wrapped estimator is already fitted. If False, the
        estimator will be fit when the visualizer is fit, otherwise, the
        estimator will not be modified. If 'auto' (default), a helper method
        will check if the estimator is fitted before fitting it again.

    show : bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however
        you cannot call ``plt.savefig`` from this signature, nor
        ``clear_figure``. If False, simply calls ``finalize()``

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Returns
    -------
    viz : SilhouetteVisualizer
        The silhouette visualizer, fitted and finalized.
    """

    oz = SilhouetteVisualizer(
        estimator, ax=ax, colors=colors, is_fitted=is_fitted, **kwargs
    )
    oz.fit(X, y)

    if show:
        oz.show()
    else:
        oz.finalize()

    return oz
