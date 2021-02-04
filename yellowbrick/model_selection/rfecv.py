# yellowbrick.model_selection.rfecv
# Visualize the number of features selected with recursive feature elimination
#
# Author:  Benjamin Bengfort
# Created: Tue Apr 03 17:31:37 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: rfecv.py [a4599db] rebeccabilbro@users.noreply.github.com $

"""
Visualize the number of features selected using recursive feature elimination
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from yellowbrick.base import ModelVisualizer
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.utils import check_X_y
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score


##########################################################################
## Recursive Feature Elimination
##########################################################################


class RFECV(ModelVisualizer):
    """
    Recursive Feature Elimination, Cross-Validated (RFECV) feature selection.

    Selects the best subset of features for the supplied estimator by removing
    0 to N features (where N is the number of features) using recursive
    feature elimination, then selecting the best subset based on the
    cross-validation score of the model. Recursive feature elimination
    eliminates n features from a model by fitting the model multiple times and
    at each step, removing the weakest features, determined by either the
    ``coef_`` or ``feature_importances_`` attribute of the fitted model.

    The visualization plots the score relative to each subset and shows trends
    in feature elimination. If the feature elimination CV score is flat, then
    potentially there are not enough features in the model. An ideal curve is
    when the score jumps from low to high as the number of features removed
    increases, then slowly decreases again from the optimal number of
    features.

    Parameters
    ----------
    estimator : a scikit-learn estimator
        An object that implements ``fit`` and provides information about the
        relative importance of features with either a ``coef_`` or
        ``feature_importances_`` attribute.

        Note that the object is cloned for each validation.

    ax : matplotlib.Axes object, optional
        The axes object to plot the figure on.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then step corresponds to the (integer)
        number of features to remove at each iteration. If within (0.0, 1.0),
        then step corresponds to the percentage (rounded down) of features to
        remove at each iteration.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        see the scikit-learn
        `cross-validation guide <http://scikit-learn.org/stable/modules/cross_validation.html>`_
        for more information on the possible strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string or scorer callable object / function with signature
        ``scorer(estimator, X, y)``. See scikit-learn model evaluation
        documentation for names of possible metrics.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    n_features_ : int
        The number of features in the selected subset

    support_ : array of shape [n_features]
        A mask of the selected features

    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranked position of feature i. Selected features are assigned rank 1.

    cv_scores_ : array of shape [n_subsets_of_features, n_splits]
        The cross-validation scores for each subset of features and splits in
        the cross-validation strategy.

    rfe_estimator_ : sklearn.feature_selection.RFE
        A fitted RFE estimator wrapping the original estimator. All estimator
        functions such as ``predict()`` and ``score()`` are passed through to
        this estimator (it rewraps the original model).

    n_feature_subsets_ : array of shape [n_subsets_of_features]
        The number of features removed on each iteration of RFE, computed by the
        number of features in the dataset and the step parameter.

    Notes
    -----
    This model wraps ``sklearn.feature_selection.RFE`` and not
    ``sklearn.feature_selection.RFECV`` because access to the internals of the
    CV and RFE estimators is required for the visualization. The visualizer
    does take similar arguments, however it does not expose the same internal
    attributes.

    Additionally, the RFE model can be accessed via the ``rfe_estimator_``
    attribute. Once fitted, the visualizer acts as a wrapper for this
    estimator and not for the original model passed to the model. This way the
    visualizer model can be used to make predictions.

    .. caution:: This visualizer requires a model that has either a ``coef_``
        or ``feature_importances_`` attribute when fitted.
    """

    def __init__(
        self, estimator, ax=None, step=1, groups=None, cv=None, scoring=None, **kwargs
    ):

        # Initialize the model visualizer
        super(RFECV, self).__init__(estimator, ax=ax, **kwargs)

        # Set parameters
        self.step = step
        self.groups = groups
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y=None):
        """
        Fits the RFECV with the wrapped model to the specified data and draws
        the rfecv curve with the optimal number of features found.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression.

        Returns
        -------
        self : instance
            Returns the instance of the RFECV visualizer.
        """
        X, y = check_X_y(X, y, "csr")
        n_features = X.shape[1]

        # This check is kind of unnecessary since RFE will do it, but it's
        # nice to get it out of the way ASAP and raise a meaningful error.
        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)

        if step <= 0:
            raise YellowbrickValueError("step must be >0")

        # Create the RFE model
        rfe = RFE(self.estimator, step=step)
        self.n_feature_subsets_ = np.arange(1, n_features + step, step)

        # Create the cross validation params
        # TODO: handle random state
        cv_params = {key: self.get_params()[key] for key in ("groups", "cv", "scoring")}

        # Perform cross-validation for each feature subset
        scores = []
        for n_features_to_select in self.n_feature_subsets_:
            rfe.set_params(n_features_to_select=n_features_to_select)
            scores.append(cross_val_score(rfe, X, y, **cv_params))

        # Convert scores to array
        self.cv_scores_ = np.array(scores)

        # Find the best RFE model
        bestidx = self.cv_scores_.mean(axis=1).argmax()
        self.n_features_ = self.n_feature_subsets_[bestidx]

        # Fit the final RFE model for the number of features
        self.rfe_estimator_ = rfe
        self.rfe_estimator_.set_params(n_features_to_select=self.n_features_)
        self.rfe_estimator_.fit(X, y)

        # Rewrap the visualizer to use the rfe estimator
        self._wrapped = self.rfe_estimator_

        # Hoist the RFE params to the visualizer
        self.support_ = self.rfe_estimator_.support_
        self.ranking_ = self.rfe_estimator_.ranking_

        self.draw()
        return self

    def draw(self, **kwargs):
        """
        Renders the rfecv curve.
        """
        # Compute the curves
        x = self.n_feature_subsets_
        means = self.cv_scores_.mean(axis=1)
        sigmas = self.cv_scores_.std(axis=1)

        # Plot one standard deviation above and below the mean
        self.ax.fill_between(x, means - sigmas, means + sigmas, alpha=0.25)

        # Plot the curve
        self.ax.plot(x, means, "o-")

        # Plot the maximum number of features
        self.ax.axvline(
            self.n_features_,
            c="k",
            ls="--",
            label="n_features = {}\nscore = {:0.3f}".format(
                self.n_features_, self.cv_scores_.mean(axis=1).max()
            ),
        )

        return self.ax

    def finalize(self, **kwargs):
        """
        Add the title, legend, and other visual final touches to the plot.
        """
        # Set the title of the figure
        self.set_title("RFECV for {}".format(self.name))

        # Add the legend
        self.ax.legend(frameon=True, loc="best")

        # Set the axis labels
        self.ax.set_xlabel("Number of Features Selected")
        self.ax.set_ylabel("Score")


##########################################################################
## Quick Methods
##########################################################################


def rfecv(
    estimator,
    X,
    y,
    ax=None,
    step=1,
    groups=None,
    cv=None,
    scoring=None,
    show=True,
    **kwargs
):
    """
    Performs recursive feature elimination with cross-validation to determine
    an optimal number of features for a model. Visualizes the feature subsets
    with respect to the cross-validation score.

    This helper function is a quick wrapper to utilize the RFECV visualizer
    for one-off analysis.

    Parameters
    ----------
    estimator : a scikit-learn estimator
        An object that implements ``fit`` and provides information about the
        relative importance of features with either a ``coef_`` or
        ``feature_importances_`` attribute.

        Note that the object is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression.

    ax : matplotlib.Axes object, optional
        The axes object to plot the figure on.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then step corresponds to the (integer)
        number of features to remove at each iteration. If within (0.0, 1.0),
        then step corresponds to the percentage (rounded down) of features to
        remove at each iteration.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        see the scikit-learn
        `cross-validation guide <http://scikit-learn.org/stable/modules/cross_validation.html>`_
        for more information on the possible strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string or scorer callable object / function with signature
        ``scorer(estimator, X, y)``. See scikit-learn model evaluation
        documentation for names of possible metrics.

    show: bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot
        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply
        calls ``finalize()``

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers. These arguments are
        also passed to the `show()` method, e.g. can pass a path to save the
        figure to.

    Returns
    -------
    viz : RFECV
        Returns the fitted, finalized visualizer.
    """
    # Initialize the visualizer
    oz = RFECV(
        estimator, ax=ax, step=step, groups=groups, cv=cv, scoring=scoring, show=show
    )

    # Fit and show the visualizer
    oz.fit(X, y)

    if show:
        oz.show()
    else:
        oz.finalize()

    # Return the visualizer object
    return oz
