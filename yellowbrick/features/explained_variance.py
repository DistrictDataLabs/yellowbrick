# yellowbrick.features.explained_variance
#
# Author:   George Richardson
# Author:   Benjamin Bengfort
# Created:  Fri Mar 2 16:16:00 2018 +0000
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: explained_variance.py [0ed6e8a] g.raymond.richardson@gmail.com $

##########################################################################
## Imports
##########################################################################

import bisect
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from yellowbrick.exceptions import NotFitted
from sklearn.preprocessing import StandardScaler
from yellowbrick.features.base import FeatureVisualizer


##########################################################################
## Explained Variance Feature Visualizer
##########################################################################


class ExplainedVariance(FeatureVisualizer):
    """

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in, the current axes
        will be used (or generated if required).

    transformer : PCA or Pipeline, default: None
        By default the visualizer creates a PCA transformer with all components, scaling
        the data on request. Users can submit their own transformer or pipeline to
        visualize the explained variance for, so long as the transformer or the last
        step in the pipeline has ``explained_variance_`` and
        ``explained_variance_ratio_`` learned attributes after being fitted.

    cumulative : bool, default: True
        Display the cumulative explained variance of components ordered by magnitude,
        otherwise display each component's direct value.

    ratio : bool, default: True
        Display the ratio of the component's explained variance to the total variance,
        otherwise display the amount of variance.

    scale : bool, default: True
        If true, the default PCA used by the visualizer has a standard scalar applied
        to the data using the mean and standard deviation. This argument is ignored if
        a user supplied transformer exists.

    n_components : int, default: None
        Whether or not to limit the number of components whose variance is explained
        in the user created transformer. This argument is ignored if a user supplied
        transformer exists.

    is_fitted : bool, default=False
        Specify if the user supplied transformer is already fitted. If False, the
        transformer will be fit when the visualizer is fit, otherwise the transformer
        will not be modified. Note that if a user supplied transformer is fitted, then
        no additional calls to the visualizer ``fit()`` method is required (a unique
        behavior of the ``ExplainedVariance`` visualizer).

    random_state : int, RandomState instance or None, optional (default None)
        Set the random state on the underlying PCA solver. Note that if a user supplied
        transformer exists, this parameter is ignored.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.
        Equal to n_components largest eigenvalues of the covariance matrix of X.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If n_components is not set then all components are stored and the sum of the
        ratios is equal to 1.0.

    Examples
    --------
    >>> visualizer = ExplainedVariance()
    >>> visualizer.fit_transform(X)
    >>> visualizer.show()

    Notes
    -----
    This visualizer wraps (by default) a sklearn.decomposition.PCA object which may have
    many other learned attributes of interest, such as ``singular_values_`` or
    ``noise_variance_``. To access these properties of the fitted underlying
    decomposition, use the visualizer's ``transformer`` property.
    """

    def __init__(
        self,
        ax=None,
        transformer=None,
        cumulative=True,
        ratio=True,
        scale=True,
        n_components=None,
        is_fitted=False,
        random_state=None,
        **kwargs
    ):
        # Initialize the visulizer
        super(ExplainedVariance, self).__init__(ax=ax, **kwargs)

        # Set the transformer and drawing parameterws
        self.cumulative = cumulative
        self.ratio = ratio
        self.scale = scale
        self.n_components = n_components
        self.is_fitted = is_fitted
        self.random_state = random_state

        # NOTE: this parameter must be set last to initialize a new transformer
        self.transformer = transformer

        # Keep track of internal state
        self._drawn_on_fit = False

    @property
    def transformer(self):
        """
        Returns the underlying transformer used for explained variance.
        """
        return self._transformer

    @transformer.setter
    def transformer(self, transformer):
        """
        Creates a PCA pipeline using scaling and number of component if None is passed
        in, otherwise sets the user supplied transformer for use in the visualization.
        """
        if transformer is None:
            # In this case we have to fit the underlying model, so ignore user
            self.is_fitted = False

            # Create either the PCA transformer if none is supplied
            transformer = PCA(
                n_components=self.n_components, random_state=self.random_state
            )

            # Add a standard scaler if specified
            if self.scale:
                transformer = Pipeline([
                    ("scale", StandardScaler(with_mean=True, with_std=True)),
                    ("pca", transformer)
                ])

        self._transformer = transformer

    def fit(self, X, y=None):
        """
        Fits the visualizer on X and transforms the data to plot it on the axes.

        Parameters
        ----------
        X : array-like of shape (n, m)
            A matrix or data frame with n instances and m features

        y : array-like of shape (n,), optional
            A vector or series with target values for each instance in X.
            Not used for ExplainedVariance but allowed here to support visual pipelines.

        Returns
        -------
        self : ExplainedVariance
            Returns the visualizer object.
        """
        if not self.is_fitted:
            self.transformer.fit(X)

        # Get the explained variance learned attributes from the transformer
        self._set_explained_variance_attributes()
        self.draw()

        # Prevent duplicate drawing on calls to fit_transform()
        self._drawn_on_fit = True
        return self

    def transform(self, X=None, y=None):
        """
        Transform the data using the underlying transformer, which usually performs
        dimensionality reduction on the imput features ``X``. This method can also be
        called with a fitted model without passing data in order to draw the explained
        variances without data.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m, optional
            A matrix of n instances with m features.

        y : ndarray or Series of length n, optional
            An array or series of target or class values.
            Not used by the transformer, but supported to allow visual pipelines.

        Returns
        -------
        Xp : ndarray or DataFrame of shape n x m
            Returns a new array-like object of transformed features of shape
            ``(len(X), self.n_components)``.
        """
        if not self._drawn_on_fit:
            # Draw on transform instead - note that this may change the attributes
            self._set_explained_variance_attributes()
            self.draw()

        if X is not None:
            return self.transformer.transform(X)
        return None

    def draw(self):
        if self.ratio:
            if not hasattr(self, "explained_variance_ratio_"):
                raise NotFitted((
                    "transformer does not have the explained_variance_ratio_, "
                    "use ratio=False or ensure the visualizer is fitted."
                ))
            X = self.explained_variance_ratio_

        else:
            if not hasattr(self, "explained_variance_"):
                raise NotFitted((
                    "transformer does not have the explained_variance_, "
                    "use ratio=True or ensure the visualizer is fitted."
                ))
            X = self.explained_variance_

        label = self.transformer.__class__.__name__
        if isinstance(self.transformer, Pipeline):
            label = self.transformer.steps[-1][1].__class__.__name__

        if self.cumulative:
            X = np.cumsum(X)
            self.ax.plot(X, label=label)

            # TODO: allow the user to specify the cutoff amounts
            prev = 0
            for cutoff in [0.0, .50, .85, .95, .999]:
                components = bisect.bisect_left(X, cutoff)
                self.ax.fill_between(np.arange(0, components), 0, X[:components], color='b', alpha=min(1-cutoff+.2, 1), label="{:0.0f}%".format(cutoff*100))

        else:
            self.ax.plot(X, label=label)

            # TODO: visualize the amount of explained variance from each component

        return self.ax

    def finalize(self, **kwargs):
        # Set the title
        title = "Explained Variance"
        if self.cumulative:
            title = "Cumulative " + title
        self.set_title(title)

        if self.ratio:
            self.ax.set_ylabel("ratio of explained variance")
        else:
            self.ax.set_ylabel("explained variance")

        self.ax.set_xlabel("number of components")
        self.ax.legend(loc="best", frameon=True)

    def _set_explained_variance_attributes(self):
        """
        Helper function to discover the required attributes on the transformer. Does
        not raise any exceptions if they cannot be found, but does not set the
        attributes on the visualizer if they aren't.
        """
        obj = self.transformer
        if isinstance(obj, Pipeline):
            obj = obj.steps[-1][1]

        for attr in ("explained_variance_", "explained_variance_ratio_"):
            if hasattr(obj, attr):
                setattr(self, attr, getattr(obj, attr))


##########################################################################
## Quick Method
##########################################################################

def explained_variance(
    X,
    y=None,
    ax=None,
    show=True,
    **kwargs
):
    """ExplainedVariance quick method.

    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features to determine principle components for.

    y : ndarray or Series of length n, default: None
        An array or series of target or class values. This argument is not used but is
        enabled for pipeline purposes.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If None is passed in, the current axes
        will be used (or generated if required).

    show : bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot
        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply
        calls ``finalize()``

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Returns
    -------
    viz : ExplainedVariance
        Returns the fitted, finalized visualizer
    """

    # Instantiate the visualizer
    oz = ExplainedVariance()

    # Fit and transform the visualizer (calls draw)
    oz.fit(X, y)
    oz.transform(X)

    if show:
        oz.show()
    else:
        oz.finalize()

    # Return the visualizer object
    return oz
