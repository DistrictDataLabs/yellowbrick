# yellowbrick.text.umap_vis
# Implements UMAP visualizations of documents in 2D space.
#
# Author:   John Healy
# Created:  Mon Dec 03 14:00:00 2018 -0500
#
# Copyright (C) 2019 The sckit-yb developers
# For license information, see LICENSE.txt
#
# ID: umap_vis.py [73a44e5] jchealy@gmail.com $

"""
Implements UMAP visualizations of documents in 2D space.
"""

##########################################################################
## Imports
##########################################################################

import warnings
import numpy as np

from collections import defaultdict

from yellowbrick.draw import manual_legend
from yellowbrick.text.base import TextVisualizer
from yellowbrick.style.colors import resolve_colors
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.pipeline import Pipeline

try:
    from umap import UMAP
except ImportError:
    UMAP = None
except (RuntimeError, AttributeError):
    UMAP = None
    warnings.warn(
        "Error Importing UMAP.  UMAP does not support python 2.7 on Windows 32 bit."
    )

##########################################################################
## Quick Methods
##########################################################################


def umap(
    X,
    y=None,
    ax=None,
    classes=None,
    colors=None,
    colormap=None,
    alpha=0.7,
    show=True,
    **kwargs
):
    """
    Display a projection of a vectorized corpus in two dimensions using UMAP (Uniform
    Manifold Approximation and Projection), a nonlinear dimensionality reduction method
    that is particularly well suited to embedding in two or three dimensions for
    visualization as a scatter plot. UMAP is a relatively new technique but is often
    used to visualize clusters or groups of data points and their relative proximities.
    It typically is fast, scalable, and can be applied directly to sparse matrices
    eliminating the need to run a ``TruncatedSVD`` as a pre-processing step.

    The current default for UMAP is Euclidean distance. Hellinger distance would be a
    more appropriate distance function to use with CountVectorize data. That will be
    released in a forthcoming version of UMAP. In the meantime cosine distance is likely
    a better text default that Euclidean and can be set using the keyword argument
    ``metric='cosine'``.

    Parameters
    ----------

    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features representing the corpus of
        vectorized documents to visualize with umap.

    y : ndarray or Series of length n
        An optional array or series of target or class values for instances.
        If this is specified, then the points will be colored according to
        their class. Often cluster labels are passed in to color the documents
        in cluster space, so this method is used both for classification and
        clustering methods.

    ax : matplotlib axes
        The axes to plot the figure on.

    classes : list of strings
        The names of the classes in the target, used to create a legend.

    colors : list or tuple of colors
        Specify the colors for each individual class

    colormap : string or matplotlib cmap
        Sequential colormap for continuous target

    alpha : float, default: 0.7
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    show : bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however
        you cannot call ``plt.savefig`` from this signature, nor
        ``clear_figure``. If False, simply calls ``finalize()``

    kwargs : dict
        Pass any additional keyword arguments to the UMAP transformer.

    -------
    visualizer: UMAPVisualizer
        Returns the fitted, finalized visualizer
    """
    # Instantiate the visualizer
    visualizer = UMAPVisualizer(
        ax=ax, classes=classes, colors=colors, colormap=colormap, alpha=alpha, **kwargs
    )

    # Fit the visualizer (calls draw)
    visualizer.fit(X, y, **kwargs)

    if show:
        visualizer.show()
    else:
        visualizer.finalize()

    # Return the visualizer object
    return visualizer


##########################################################################
## UMAPVisualizer
##########################################################################


class UMAPVisualizer(TextVisualizer):
    """
    Display a projection of a vectorized corpus in two dimensions using UMAP (Uniform
    Manifold Approximation and Projection), a nonlinear dimensionality reduction method
    that is particularly well suited to embedding in two or three dimensions for
    visualization as a scatter plot. UMAP is a relatively new technique but is often
    used to visualize clusters or groups of data points and their relative proximities.
    It typically is fast, scalable, and can be applied directly to sparse matrices
    eliminating the need to run a ``TruncatedSVD`` as a pre-processing step.

    The current default for UMAP is Euclidean distance. Hellinger distance would be a
    more appropriate distance function to use with CountVectorize data. That will be
    released in a forthcoming version of UMAP. In the meantime cosine distance is likely
    a better text default that Euclidean and can be set using the keyword argument
    ``metric='cosine'``.

    For more, see https://github.com/lmcinnes/umap

    Parameters
    ----------

    ax : matplotlib axes
        The axes to plot the figure on.

    labels : list of strings
        The names of the classes in the target, used to create a legend.
        Labels must match names of classes in sorted order.

    colors : list or tuple of colors
        Specify the colors for each individual class

    colormap : string or matplotlib cmap
        Sequential colormap for continuous target

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random. The random state is applied to the preliminary
        decomposition as well as UMAP.

    alpha : float, default: 0.7
        Specify a transparency where 1 is completely opaque and 0 is completely
        transparent. This property makes densely clustered points more visible.

    kwargs : dict
        Pass any additional keyword arguments to the UMAP transformer.

    Examples
    --------

    >>> model = MyVisualizer(metric='cosine')
    >>> model.fit(X)
    >>> model.show()

    """

    # NOTE: cannot be np.nan
    NULL_CLASS = None

    def __init__(
        self,
        ax=None,
        labels=None,
        classes=None,
        colors=None,
        colormap=None,
        random_state=None,
        alpha=0.7,
        **kwargs
    ):

        if UMAP is None:
            raise YellowbrickValueError(
                (
                    "umap package doesn't seem to be installed."
                    "Please install UMAP via: pip install umap-learn"
                )
            )

        # Visual Parameters
        self.alpha = alpha
        self.labels = labels
        self.colors = colors
        self.colormap = colormap
        self.random_state = random_state

        # Fetch UMAP kwargs from kwargs by popping only keys belonging to UMAP params
        umap_kwargs = {
            key: kwargs.pop(key) for key in UMAP().get_params() if key in kwargs
        }

        # UMAP doesn't require any pre-processing before embedding and thus doesn't
        # require a pipeline.
        self.transformer_ = self.make_transformer(umap_kwargs)

        # Call super at the end so that size and title are set correctly
        super(UMAPVisualizer, self).__init__(ax=ax, **kwargs)

    def make_transformer(self, umap_kwargs={}):
        """
        Creates an internal transformer pipeline to project the data set into
        2D space using UMAP. This method will reset the transformer on the
        class.

        Parameters
        ----------
        umap_kwargs : dict
            Keyword arguments for the internal UMAP transformer

        Returns
        -------
        transformer : Pipeline
            Pipelined transformer for UMAP projections
        """

        # Create the pipeline steps
        steps = []

        # Add the UMAP manifold
        steps.append(
            (
                "umap",
                UMAP(n_components=2, random_state=self.random_state, **umap_kwargs),
            )
        )

        # return the pipeline
        return Pipeline(steps)

    def fit(self, X, y=None, **kwargs):
        """
        The fit method is the primary drawing input for the UMAP projection
        since the visualization requires both X and an optional y value. The
        fit method expects an array of numeric vectors, so text documents must
        be vectorized before passing them to this method.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features representing the corpus of
            vectorized documents to visualize with UMAP.

        y : ndarray or Series of length n
            An optional array or series of target or class values for
            instances. If this is specified, then the points will be colored
            according to their class. Often cluster labels are passed in to
            color the documents in cluster space, so this method is used both
            for classification and clustering methods.

        kwargs : dict
            Pass generic arguments to the drawing method

        Returns
        -------
        self : instance
            Returns the instance of the transformer/visualizer
        """

        # Store the classes we observed in y
        if y is not None:
            self.classes_ = np.unique(y)
        elif y is None and self.labels is not None:
            self.classes_ = np.array([self.labels[0]])
        else:
            self.classes_ = np.array([self.NULL_CLASS])

        # Fit our internal transformer and transform the data.
        vecs = self.transformer_.fit_transform(X)
        self.n_instances_ = vecs.shape[0]

        # Draw the vectors
        self.draw(vecs, y, **kwargs)

        # Fit always returns self.
        return self

    def draw(self, points, target=None, **kwargs):
        """
        Called from the fit method, this method draws the UMAP scatter plot,
        from a set of decomposed points in 2 dimensions. This method also
        accepts a third dimension, target, which is used to specify the colors
        of each of the points. If the target is not specified, then the points
        are plotted as a single cloud to show similar documents.
        """
        # Resolve the labels with the classes
        labels = self.labels if self.labels is not None else self.classes_
        if len(labels) != len(self.classes_):
            raise YellowbrickValueError(
                (
                    "number of supplied labels ({}) does not "
                    "match the number of classes ({})"
                ).format(len(labels), len(self.classes_))
            )

        # Create the color mapping for the labels.
        self.color_values_ = resolve_colors(
            n_colors=len(labels), colormap=self.colormap, colors=self.colors
        )
        colors = dict(zip(labels, self.color_values_))

        # Transform labels into a map of class to label
        labels = dict(zip(self.classes_, labels))

        # Expand the points into vectors of x and y for scatter plotting,
        # assigning them to their label if the label has been passed in.
        # Additionally, filter classes not specified directly by the user.
        series = defaultdict(lambda: {"x": [], "y": []})

        if target is not None:
            for t, point in zip(target, points):
                label = labels[t]
                series[label]["x"].append(point[0])
                series[label]["y"].append(point[1])
        else:
            label = self.classes_[0]
            for x, y in points:
                series[label]["x"].append(x)
                series[label]["y"].append(y)

        # Plot the points
        for label, points in series.items():
            self.ax.scatter(
                points["x"], points["y"], c=colors[label], alpha=self.alpha, label=label
            )

        return self.ax

    def finalize(self, **kwargs):
        """
        Finalize the drawing by adding a title and legend, and removing the
        axes objects that do not convey information about UMAP.
        """
        self.set_title("UMAP Projection of {} Documents".format(self.n_instances_))

        # Remove the ticks
        self.ax.set_yticks([])
        self.ax.set_xticks([])

        # Add the legend outside of the figure box.
        if not all(self.classes_ == np.array([self.NULL_CLASS])):
            box = self.ax.get_position()
            self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            manual_legend(
                self,
                self.classes_,
                self.color_values_,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )
