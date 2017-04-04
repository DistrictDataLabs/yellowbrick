# yellowbrick.cluster.elbow
# Implements the elbow method for determining the optimal number of clusters.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Mar 23 22:36:31 2017 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: elbow.py [] benjamin@bengfort.com $

"""
Implements the elbow method for determining the optimal number of clusters.
https://bl.ocks.org/rpgove/0060ff3b656618e9136b
"""

##########################################################################
## Imports
##########################################################################

from .base import ClusteringScoreVisualizer
from ..exceptions import YellowbrickValueError

from sklearn.metrics import silhouette_score

## Packages for export
__all__ = [
    "KElbowVisualizer",
]


##########################################################################
## Elbow Method
##########################################################################

class KElbowVisualizer(ClusteringScoreVisualizer):
    """
    The K-Elbow Visualizer implements the "elbow" method of selecting the
    optimal number of clusters for K-means clustering. K-means is a simple
    unsupervised machine learning algorithm that groups data into a specified
    number (k) of clusters. Because the user must specify in advance what k to
    choose, the algorithm is somewhat naive -- it assigns all members to k
    clusters even if that is not the right k for the dataset.

    The elbow method runs k-means clustering on the dataset for a range of
    values for k (say from 1-10) and then for each value of k computes the
    ``silhouette_score``, the mean silhouette coefficient for all samples.
    The silhouette coefficient of a cluster is computed by comparing the mean
    intra-cluster distance (a) and the mean nearest-cluster distance (b) for
    each sample in the dataset; the silhouette is then computed as
    ``(b-a) / max(a,b)``. The score is a value between -1 and 1, values near
    zero indicate overlapping clusters. Negative values imply that samples
    have been assigned to the wrong cluster, and positive values mean that
    there are discrete clusters.

    Finally, the silhouette score for each k is plotted. If the line chart
    looks like an arm, then the "elbow" (the point of inflection on the curve)
    is the best value of k. The idea is that we want as small a k as possible
    such that the clusters do not overlap.

    Parameters
    ----------

    model : a Scikit-Learn clusterer
        Should be an instance of a clusterer, specifically ``KMeans`` or
        ``MiniBatchKMeans``. If it is not a clusterer, an exception is raised.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    k : integer or tuple
        The range of k to compute silhouette scores for. If a single integer
        is specified, then will compute the range (2,k) otherwise the
        specified range in the tuple is used.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Examples
    --------

    >>> from yellowbrick.cluster import KElbowVisualizer
    >>> from sklearn.cluster import KMeans
    >>> model = KElbowVisualizer(KMeans(), k=10)
    >>> model.fit(X)
    >>> model.poof()

    Notes
    -----

    If you get a visualizer that doesn't have an elbow or inflection point,
    then this method may not be working. The elbow method does not work well
    if the data is not very clustered; in this case you might see a smooth
    curve and the value of k is unclear. Other scoring methods such as BIC or
    SSE also can be used to explore if clustering is a correct choice.

    For a discussion on the Elbow method, read more at
    `Robert Gove's Block <https://bl.ocks.org/rpgove/0060ff3b656618e9136b>`_.

    .. todo:: add parallelization ooption for performance
    .. todo:: add different metrics for scores and silhoutte
    .. todo:: add timing information about how long its taking
    """

    def __init__(self, model, ax=None, k=10, **kwargs):
        super(KElbowVisualizer, self).__init__(model, ax=ax, **kwargs)

        # Convert K into a tuple argument if an integer
        if isinstance(k, int):
            k = (2, k)

        # Expand k in to the values we will use, capturing exceptions
        try:
            k = tuple(k)
            self.k_values_ = list(range(*k))
        except:
            raise YellowbrickValueError((
                "Specify a range or maximal K value, the value '{}' "
                "is not a valid argument for K.".format(k)
            ))


        # Holds the values of the silhoutte scores
        self.k_scores_ = None

    def fit(self, X, y=None, **kwargs):
        """
        Fits n KMeans models where n is the length of ``self.k_values_``,
        storing the silhoutte scores in the ``self.k_scores_`` attribute.
        This method finishes up by calling draw to create the plot.
        """

        self.k_scores_ = []

        for k in self.k_values_:
            self.estimator.set_params(n_clusters=k)
            self.estimator.fit(X)
            self.k_scores_.append(
                silhouette_score(X, self.estimator.labels_, metric='euclidean')
            )

        self.draw()

    def draw(self):
        """
        Draw the elbow curve for the specified scores and values of K.
        """

        if self.ax is None:
            self.ax = self.gca()

        # Plot the silhouette score against k
        self.ax.plot(self.k_values_, self.k_scores_)

        return self.ax

    def finalize(self):
        """
        Prepare the figure for rendering by setting the title as well as the
        X and Y axis labels and adding the legend.
        """

        # Set the title
        self.set_title(
            'Silhoutte for {} Clustering Behavior'.format(self.name)
        )

        # Set the x and y labels
        self.ax.set_xlabel('k')
        self.ax.set_ylabel('silhouette')
