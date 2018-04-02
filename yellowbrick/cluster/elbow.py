# yellowbrick.cluster.elbow
# Implements the elbow method for determining the optimal number of clusters.
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Mar 23 22:36:31 2017 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: elbow.py [5a370c8] benjamin@bengfort.com $

"""
Implements the elbow method for determining the optimal number of clusters.
https://bl.ocks.org/rpgove/0060ff3b656618e9136b
"""

##########################################################################
## Imports
##########################################################################

import time

from .base import ClusteringScoreVisualizer
from ..exceptions import YellowbrickValueError

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder


## Packages for export
__all__ = [
    "KElbowVisualizer", "distortion_score"
]


##########################################################################
## Metrics
##########################################################################

def distortion_score(X, labels, metric='euclidean'):
    """
    Compute the mean distortion of all samples.

    The distortion is computed as the the sum of the squared distances between
    each observation and its closest centroid. Logically, this is the metric
    that K-Means attempts to minimize as it is fitting the model.

    .. seealso:: http://kldavenport.com/the-cost-function-of-k-means/

    Parameters
    ----------
    X : array, shape = [n_samples, n_features] or [n_samples_a, n_samples_a]
        Array of pairwise distances between samples if metric == "precomputed"
        or a feature array for computing distances against the labels.

    labels : array, shape = [n_samples]
        Predicted labels for each sample

    metric : string
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by `sklearn.metrics.pairwise.pairwise_distances
        <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html#sklearn.metrics.pairwise.pairwise_distances>`_

    .. todo:: add sample_size and random_state kwds similar to silhouette_score
    """
    # Encode labels to get unique centers and groups
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    unique_labels = le.classes_

    # Sum of the distortions
    distortion = 0

    # Loop through each label (center) to compute the centroid
    for current_label in unique_labels:
        # Mask the instances that belong to the current label
        mask = labels == current_label
        instances = X[mask]

        # Compute the center of these instances
        center = instances.mean(axis=0)

        # Compute the square distances from the instances to the center
        distances = pairwise_distances(instances, [center], metric=metric)
        distances = distances ** 2

        # Add the mean square distance to the distortion
        distortion += distances.mean()

    return distortion


##########################################################################
## Elbow Method
##########################################################################

KELBOW_SCOREMAP = {
    "distortion": distortion_score,
    "silhouette": silhouette_score,
    "calinski_harabaz": calinski_harabaz_score,
}


class KElbowVisualizer(ClusteringScoreVisualizer):
    """
    The K-Elbow Visualizer implements the "elbow" method of selecting the
    optimal number of clusters for K-means clustering. K-means is a simple
    unsupervised machine learning algorithm that groups data into a specified
    number (k) of clusters. Because the user must specify in advance what k to
    choose, the algorithm is somewhat naive -- it assigns all members to k
    clusters even if that is not the right k for the dataset.

    The elbow method runs k-means clustering on the dataset for a range of
    values for k (say from 1-10) and then for each value of k computes an
    average score for all clusters. By default, the ``distortion_score`` is
    computed, the sum of square distances from each point to its assigned
    center. Other metrics can also be used such as the ``silhouette_score``,
    the mean silhouette  coefficient for all samples or the
    ``calinski_harabaz_score``, which computes the ratio of dispersion between
    and within clusters.

    When these overall metrics for each model are plotted, it is possible to
    visually determine the best value for K. If the line chart looks like an
    arm, then the "elbow" (the point of inflection on the curve) is the best
    value of k. The "arm" can be either up or down, but if there is a strong
    inflection point, it is a good indication that the underlying model fits
    best at that point.

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

    metric : string, default: ``"distortion"``
        Select the scoring metric to evaluate the clusters. The default is the
        mean distortion, defined by the sum of squared distances between each
        observation and its closest centroid. Other metrics include:

        - **distortion**: mean sum of squared distances to centers
        - **silhouette**: mean ratio of intra-cluster and nearest-cluster distance
        - **calinski_harabaz**: ratio of within to between cluster dispersion

    timings : bool, default: True
        Display the fitting time per k to evaluate the amount of time required
        to train the clustering model.

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

    .. todo:: add parallelization option for performance
    .. todo:: add different metrics for scores and silhoutte
    .. todo:: add timing information about how long its taking
    """

    def __init__(self, model, ax=None, k=10,
                 metric="distortion", timings=True, **kwargs):
        super(KElbowVisualizer, self).__init__(model, ax=ax, **kwargs)

        # Get the scoring method
        if metric not in KELBOW_SCOREMAP:
            raise YellowbrickValueError(
                "'{}' is not a defined metric "
                "use one of distortion, silhouette, or calinski_harabaz"
            )

        # Store the arguments
        self.scoring_metric = KELBOW_SCOREMAP[metric]
        self.timings = timings

        # Convert K into a tuple argument if an integer
        if isinstance(k, int):
            k = (2, k+1)

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
        self.k_timers_ = []

        for k in self.k_values_:
            # Compute the start time for each  model
            start = time.time()

            # Set the k value and fit the model
            self.estimator.set_params(n_clusters=k)
            self.estimator.fit(X)

            # Append the time and score to our plottable metrics
            self.k_timers_.append(time.time() - start)
            self.k_scores_.append(
                self.scoring_metric(X, self.estimator.labels_)
            )

        self.draw()

        return self

    def draw(self):
        """
        Draw the elbow curve for the specified scores and values of K.
        """
        # Plot the silhouette score against k
        self.ax.plot(self.k_values_, self.k_scores_, marker="D", label="score")

        # If we're going to plot the timings, create a twinx axis
        if self.timings:
            self.axes = [self.ax, self.ax.twinx()]
            self.axes[1].plot(
                self.k_values_, self.k_timers_, label="fit time",
                c='g', marker="o", linestyle="--", alpha=0.75,
            )

        return self.ax

    def finalize(self):
        """
        Prepare the figure for rendering by setting the title as well as the
        X and Y axis labels and adding the legend.
        """
        # Get the metric name
        metric = self.scoring_metric.__name__.replace("_", " ").title()

        # Set the title
        self.set_title(
            '{} Elbow for {} Clustering'.format(metric, self.name)
        )

        # Set the x and y labels
        self.ax.set_xlabel('k')
        self.ax.set_ylabel(metric.lower())

        # Set the second y axis labels
        if self.timings:
            self.axes[1].grid(False)
            self.axes[1].set_ylabel("fit time (seconds)", color='g')
            self.axes[1].tick_params('y', colors='g')
