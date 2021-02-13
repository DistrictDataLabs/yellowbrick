# yellowbrick.cluster.icdm
# Implements Intercluster Distance Map visualizations.
#
# Author:  Benjamin Bengfort
# Created: Tue Aug 21 11:56:53 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: icdm.py [2f23976] benjamin@bengfort.com $

"""
Implements Intercluster Distance Map visualizations.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from sklearn.manifold import MDS, TSNE

from yellowbrick.utils.timer import Timer
from yellowbrick.utils.decorators import memoized
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.cluster.base import ClusteringScoreVisualizer
from yellowbrick.utils.helpers import prop_to_size, check_fitted

try:
    # Only available in Matplotlib >= 2.0.2
    from mpl_toolkits.axes_grid1 import inset_locator
except ImportError:
    inset_locator = None


## Packages for export
__all__ = [
    "InterclusterDistance",
    "intercluster_distance",
    "VALID_EMBEDDING",
    "VALID_SCORING",
    "ICDM",
]


# Valid strings to use for embedding names
VALID_EMBEDDING = {"mds", "tsne"}

# Valid strings to use for scoring names
VALID_SCORING = {"membership"}


##########################################################################
## InterclusterDistance Visualizer
##########################################################################


class InterclusterDistance(ClusteringScoreVisualizer):
    """
    Intercluster distance maps display an embedding of the cluster centers in
    2 dimensions with the distance to other centers preserved. E.g. the closer
    to centers are in the visualization, the closer they are in the original
    feature space. The clusters are sized according to a scoring metric. By
    default, they are sized by membership, e.g. the number of instances that
    belong to each center. This gives a sense of the relative importance of
    clusters. Note however, that because two clusters overlap in the 2D space,
    it does not imply that they overlap in the original feature space.

    Parameters
    ----------
    estimator : a Scikit-Learn clusterer
        Should be an instance of a centroidal clustering algorithm (or a
        hierarchical algorithm with a specified number of clusters). Also
        accepts some other models like LDA for text clustering.
        If it is not a clusterer, an exception is raised. If the estimator
        is not fitted, it is fit when the visualizer is fitted, unless
        otherwise specified by ``is_fitted``.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    min_size : int, default: 400
        The size, in points, of the smallest cluster drawn on the graph.
        Cluster sizes will be scaled between the min and max sizes.

    max_size : int, default: 25000
        The size, in points, of the largest cluster drawn on the graph.
        Cluster sizes will be scaled between the min and max sizes.

    embedding : default: 'mds'
        The algorithm used to embed the cluster centers in 2 dimensional space
        so that the distance between clusters is represented equivalently to
        their relationship in feature spaceself.
        Embedding algorithm options include:

        - **mds**: multidimensional scaling
        - **tsne**: stochastic neighbor embedding

    scoring : default: 'membership'
        The scoring method used to determine the size of the clusters drawn on
        the graph so that the relative importance of clusters can be viewed.
        Scoring method options include:

        - **membership**: number of instances belonging to each cluster

    legend : bool, default: True
        Whether or not to draw the size legend onto the graph, omit the legend
        to more easily see clusters that overlap.

    legend_loc : str, default: "lower left"
        The location of the legend on the graph, used to move the legend out
        of the way of clusters into open space. The same legend location
        options for matplotlib are used here.

        .. seealso:: https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend

    legend_size : float, default: 1.5
        The size, in inches, of the size legend to inset into the graph.

    random_state : int or RandomState, default: None
        Fixes the random state for stochastic embedding algorithms.

    is_fitted : bool or str, default='auto'
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If 'auto' (default), a helper method will check if the estimator
        is fitted before fitting it again.

    kwargs : dict
        Keyword arguments passed to the base class and may influence the
        feature visualization properties.

    Attributes
    ----------
    cluster_centers_ : array of shape (n_clusters, n_features)
        The computed cluster centers from the underlying model.

    embedded_centers_ : array of shape (n_clusters, 2)
        The positions of all the cluster centers on the graph.

    scores_ : array of shape (n_clusters,)
        The scores of each cluster that determine its size on the graph.

    fit_time_ : Timer
        The time it took to fit the clustering model and perform the embedding.

    Notes
    -----
    Currently the only two embeddings supported are MDS and TSNE. Soon to
    follow will be PCoA and a customized version of PCoA for LDA. The only
    supported scoring metric is membership, but in the future, silhouette
    scores and cluster diameter will be added.

    In terms of algorithm support, right now any clustering algorithm that has
    a learned ``cluster_centers_`` and ``labels_`` attribute will work with
    the visualizer. In the future, we will update this to work with hierarchical
    clusterers that have ``n_components`` and LDA.
    """

    def __init__(
        self,
        estimator,
        ax=None,
        min_size=400,
        max_size=25000,
        embedding="mds",
        scoring="membership",
        legend=True,
        legend_loc="lower left",
        legend_size=1.5,
        random_state=None,
        is_fitted="auto",
        **kwargs
    ):
        # Initialize the visualizer bases
        super(InterclusterDistance, self).__init__(estimator, ax=ax, **kwargs)

        # Ensure that a valid embedding and scoring is passed in
        validate_embedding(embedding)
        validate_scoring(scoring)

        # Set decomposition properties
        self.scoring = scoring
        self.embedding = embedding
        self.random_state = random_state

        # Set visual properties
        self.legend = legend
        self.min_size = min_size
        self.max_size = max_size
        self.legend_loc = legend_loc
        self.legend_size = legend_size

        # Colors are currently hardcoded, need to compute face and edge color
        # from this color based on the alpha of the cluster center. The user
        # can "hack" these properties before drawing, however.
        self.facecolor = "#2e719344"
        self.edgecolor = "#2e719399"

        if self.legend:
            self.lax  # If legend True, test the version availability

    @memoized
    def lax(self):
        """
        Returns the legend axes, creating it only on demand by creating a 2"
        by 2" inset axes that has no grid, ticks, spines or face frame (e.g
        is mostly invisible). The legend can then be drawn on this axes.
        """
        if inset_locator is None:
            raise YellowbrickValueError(
                (
                    "intercluster distance map legend requires matplotlib 2.0.2 or "
                    "later please upgrade matplotlib or set legend=False "
                )
            )

        lax = inset_locator.inset_axes(
            self.ax,
            width=self.legend_size,
            height=self.legend_size,
            loc=self.legend_loc,
        )

        lax.set_frame_on(False)
        lax.set_facecolor("none")
        lax.grid(False)
        lax.set_xlim(-1.4, 1.4)
        lax.set_ylim(-1.4, 1.4)
        lax.set_xticks([])
        lax.set_yticks([])

        for name in lax.spines:
            lax.spines[name].set_visible(False)

        return lax

    @memoized
    def transformer(self):
        """
        Creates the internal transformer that maps the cluster center's high
        dimensional space to its two dimensional space.
        """
        ttype = self.embedding.lower()  # transformer method type

        if ttype == "mds":
            return MDS(n_components=2, random_state=self.random_state)

        if ttype == "tsne":
            return TSNE(n_components=2, random_state=self.random_state)

        raise YellowbrickValueError("unknown embedding '{}'".format(ttype))

    @property
    def cluster_centers_(self):
        """
        Searches for or creates cluster centers for the specified clustering
        algorithm. This algorithm ensures that that the centers are
        appropriately drawn and scaled so that distance between clusters are
        maintained.
        """
        # TODO: Handle agglomerative clustering and LDA
        for attr in ("cluster_centers_",):
            try:
                return getattr(self.estimator, attr)
            except AttributeError:
                continue

        raise AttributeError(
            "could not find or make cluster_centers_ for {}".format(
                self.estimator.__class__.__name__
            )
        )

    def fit(self, X, y=None):
        """
        Fit the clustering model, computing the centers then embeds the centers
        into 2D space using the embedding method specified.
        """
        with Timer() as self.fit_time_:
            if not check_fitted(self.estimator, is_fitted_by=self.is_fitted):
                # Fit the underlying estimator
                self.estimator.fit(X, y)

        # Get the centers
        # TODO: is this how sklearn stores all centers in the model?
        C = self.cluster_centers_

        # Embed the centers in 2D space and get the cluster scores
        self.embedded_centers_ = self.transformer.fit_transform(C)
        self.scores_ = self._score_clusters(X, y)

        # Draw the clusters and fit returns self
        self.draw()
        return self

    def draw(self):
        """
        Draw the embedded centers with their sizes on the visualization.
        """
        # Compute the sizes of the markers from their score
        sizes = self._get_cluster_sizes()

        # Draw the scatter plots with associated sizes on the graph
        self.ax.scatter(
            self.embedded_centers_[:, 0],
            self.embedded_centers_[:, 1],
            s=sizes,
            c=self.facecolor,
            edgecolor=self.edgecolor,
            linewidth=1,
        )

        # Annotate the clusters with their labels
        for i, pt in enumerate(self.embedded_centers_):
            self.ax.text(
                s=str(i), x=pt[0], y=pt[1], va="center", ha="center", fontweight="bold"
            )

        # Ensure the current axes is always the main residuals axes
        plt.sca(self.ax)
        return self.ax

    def finalize(self):
        """
        Finalize the visualization to create an "origin grid" feel instead of
        the default matplotlib feel. Set the title, remove spines, and label
        the grid with components. This function also adds a legend from the
        sizes if required.
        """
        # Set the default title if a user hasn't supplied one
        self.set_title(
            "{} Intercluster Distance Map (via {})".format(
                self.estimator.__class__.__name__, self.embedding.upper()
            )
        )

        # Create the origin grid and minimalist display
        self.ax.set_xticks([0])
        self.ax.set_yticks([0])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_xlabel("PC2")
        self.ax.set_ylabel("PC1")

        # Make the legend by creating an inset axes that shows relative sizing
        # based on the scoring metric supplied by the user.
        if self.legend:
            self._make_size_legend()

    def _score_clusters(self, X, y=None):
        """
        Determines the "scores" of the cluster, the metric that determines the
        size of the cluster visualized on the visualization.
        """
        stype = self.scoring.lower()  # scoring method name

        if stype == "membership":
            return np.bincount(self.estimator.labels_)

        raise YellowbrickValueError("unknown scoring method '{}'".format(stype))

    def _get_cluster_sizes(self):
        """
        Returns the marker size (in points, e.g. area of the circle) based on
        the scores, using the prop_to_size scaling mechanism.
        """
        # NOTE: log and power are hardcoded, should we allow the user to specify?
        return prop_to_size(
            self.scores_, mi=self.min_size, ma=self.max_size, log=False, power=0.5
        )

    def _make_size_legend(self):
        """
        Draw a legend that shows relative sizes of the clusters at the 25th,
        50th, and 75th percentile based on the current scoring metric.
        """
        # Compute the size of the markers and scale them to our figure size
        # NOTE: the marker size is the area of the plot, we need to compute the
        # radius of the markers.
        areas = self._get_cluster_sizes()
        radii = np.sqrt(areas / np.pi)
        scaled = np.interp(radii, (radii.min(), radii.max()), (0.1, 1))

        # Compute the locations of the 25th, 50th, and 75th percentile scores
        indices = np.array([percentile_index(self.scores_, p) for p in (25, 50, 75)])

        # Draw size circles annotated with the percentile score as the legend.
        for idx in indices:
            # TODO: should the size circle's center be hard coded like this?
            center = (-0.30, 1 - scaled[idx])
            c = Circle(
                center,
                scaled[idx],
                facecolor="none",
                edgecolor="#2e7193",
                linewidth=1.5,
                linestyle="--",
            )
            self.lax.add_patch(c)

            # Add annotation to the size circle with the value of the score
            self.lax.annotate(
                self.scores_[idx],
                (-0.30, 1 - (2 * scaled[idx])),
                xytext=(1, 1 - (2 * scaled[idx])),
                arrowprops=dict(arrowstyle="wedge", color="#2e7193"),
                va="center",
                ha="center",
            )

        # Draw size legend title
        self.lax.text(s="membership", x=0, y=1.2, va="center", ha="center")

        # Ensure the current axes is always the main axes after modifying the
        # inset axes and while drawing.
        plt.sca(self.ax)


# alias
ICDM = InterclusterDistance

##########################################################################
## Helper Methods
##########################################################################


def percentile_index(a, q):
    """
    Returns the index of the value at the Qth percentile in array a.
    """
    return np.where(a == np.percentile(a, q, interpolation="nearest"))[0][0]


def validate_string_param(s, valid, param_name="param"):
    """
    Raises a well formatted exception if s is not in valid, otherwise does not
    raise an exception. Uses ``param_name`` to identify the parameter.
    """
    if s.lower() not in valid:
        raise YellowbrickValueError(
            "unknown {} '{}', chose from '{}'".format(param_name, s, ", ".join(valid))
        )


def validate_embedding(param):
    """
    Raises an exception if the param is not in VALID_EMBEDDING
    """
    validate_string_param(param, VALID_EMBEDDING, "embedding")


def validate_scoring(param):
    """
    Raises an exception if the param is not in VALID_SCORING
    """
    validate_string_param(param, VALID_SCORING, "scoring")


##########################################################################
## Quick Method
##########################################################################


def intercluster_distance(
    estimator,
    X,
    y=None,
    ax=None,
    min_size=400,
    max_size=25000,
    embedding="mds",
    scoring="membership",
    legend=True,
    legend_loc="lower left",
    legend_size=1.5,
    random_state=None,
    is_fitted="auto",
    show=True,
    **kwargs
):
    """Quick Method:
    Intercluster distance maps display an embedding of the cluster centers in
    2 dimensions with the distance to other centers preserved. E.g. the closer
    to centers are in the visualization, the closer they are in the original
    feature space. The clusters are sized according to a scoring metric. By
    default, they are sized by membership, e.g. the number of instances that
    belong to each center. This gives a sense of the relative importance of
    clusters. Note however, that because two clusters overlap in the 2D space,
    it does not imply that they overlap in the original feature space.

    Parameters
    ----------
    estimator : a Scikit-Learn clusterer
        Should be an instance of a centroidal clustering algorithm (or a
        hierarchical algorithm with a specified number of clusters). Also
        accepts some other models like LDA for text clustering.
        If it is not a clusterer, an exception is raised. If the estimator
        is not fitted, it is fit when the visualizer is fitted, unless
        otherwise specified by ``is_fitted``.

    X : array-like of shape (n, m)
        A matrix or data frame with n instances and m features

    y : array-like of shape (n,), optional
        A vector or series representing the target for each instance

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    min_size : int, default: 400
        The size, in points, of the smallest cluster drawn on the graph.
        Cluster sizes will be scaled between the min and max sizes.

    max_size : int, default: 25000
        The size, in points, of the largest cluster drawn on the graph.
        Cluster sizes will be scaled between the min and max sizes.

    embedding : default: 'mds'
        The algorithm used to embed the cluster centers in 2 dimensional space
        so that the distance between clusters is represented equivalently to
        their relationship in feature spaceself.
        Embedding algorithm options include:

        - **mds**: multidimensional scaling
        - **tsne**: stochastic neighbor embedding

    scoring : default: 'membership'
        The scoring method used to determine the size of the clusters drawn on
        the graph so that the relative importance of clusters can be viewed.
        Scoring method options include:

        - **membership**: number of instances belonging to each cluster

    legend : bool, default: True
        Whether or not to draw the size legend onto the graph, omit the legend
        to more easily see clusters that overlap.

    legend_loc : str, default: "lower left"
        The location of the legend on the graph, used to move the legend out
        of the way of clusters into open space. The same legend location
        options for matplotlib are used here.

        .. seealso:: https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend

    legend_size : float, default: 1.5
        The size, in inches, of the size legend to inset into the graph.

    random_state : int or RandomState, default: None
        Fixes the random state for stochastic embedding algorithms.

    is_fitted : bool or str, default='auto'
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If 'auto' (default), a helper method will check if the estimator
        is fitted before fitting it again.

    show : bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however
        you cannot call ``plt.savefig`` from this signature, nor
        ``clear_figure``. If False, simply calls ``finalize()``

    kwargs : dict
        Keyword arguments passed to the base class and may influence the
        feature visualization properties.

    Returns
    -------
    viz : InterclusterDistance
        The intercluster distance visualizer, fitted and finalized.
    """
    oz = InterclusterDistance(
        estimator,
        ax=ax,
        min_size=min_size,
        max_size=max_size,
        embedding=embedding,
        scoring=scoring,
        legend=legend,
        legend_loc=legend_loc,
        legend_size=legend_size,
        random_state=random_state,
        is_fitted=is_fitted,
        **kwargs
    )

    oz.fit(X, y)

    if show:
        oz.show()
    else:
        oz.finalize()

    return oz
