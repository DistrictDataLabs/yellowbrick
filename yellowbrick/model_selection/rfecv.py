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

# TODO: does this require a minimum sklearn version?
from sklearn.utils import check_X_y
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.base import is_classifier, clone
from sklearn.feature_selection._rfe import RFECV as skRFECV
from sklearn.feature_selection._rfe import RFE, _rfe_single_fit

try:
    # TODO: do we need to make joblib an optional dependency?
    from joblib import Parallel, delayed, effective_n_jobs
except ImportError:
    Parallel, delayed = None, None

    def effective_n_jobs(*args, **kwargs):
        return 1


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
    model : a scikit-learn estimator
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

    min_features_to_select : int (default=1)
        The minimum number of features to be selected. This number of features will
        always be scored, even if the difference between the original feature count and
        min_features_to_select isn’t divisible by step.

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

    verbose : int, default: 0
        Controls verbosity of output.

    n_jobs : int or None, optional (default=None)
        Number of cores to run in parallel while fitting across folds. None means 1
        unless in a joblib.parallel_backend context. -1 means using all processors.

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

    grid_scores_ : array of shape [n_subsets_of_features]
        The cross-validation scores such that grid_scores_[i] corresponds to the CV
        score of the i-th subset of features.

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
        self, model, ax=None, step=1, groups=None, cv=None, scoring=None, min_features_to_select=1, **kwargs
    ):

        # Initialize the model visualizer
        super(RFECV, self).__init__(model, ax=ax, **kwargs)

        # Set parameters
        # TODO: update these parameters
        self.set_params(step=step, groups=groups, cv=cv, scoring=scoring, min_features_to_select=min_features_to_select)

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
        # Create and fit the RFECV model
        self.rfe_estimator_ = _RFECV(self.estimator)
        self.rfe_estimator_.set_params(**self.get_rfecv_params())
        self.rfe_estimator_.fit(X, y, groups=self.groups)

        # HACK: this is wrong and needs to be fixed
        n_features = X.shape[1]
        step = int(self.step)
        self.n_feature_subsets_ = np.arange(1, np.ceil((n_features - self.min_features_to_select) / step) + 1)

        # Modify the internal estimator to be the final fitted estimator
        self._wrapped = self.rfe_estimator_.estimator_

        # Hoist the RFE params to the visualizer
        for attr in ("cv_scores_", "n_features_", "support_", "ranking_", "grid_scores_"):
            setattr(self, attr, getattr(self.rfe_estimator_, attr))

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

    def get_rfecv_params(self):
        params = self.get_params()
        for param in ("model", "ax", "kwargs", "groups"):
            if param in params:
                del params[param]
        return params


##########################################################################
## Quick Methods
##########################################################################

# TODO: update the quick method params
def rfecv(
    model, X, y,
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
    model : a scikit-learn estimator
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
        model, ax=ax, step=step, groups=groups, cv=cv, scoring=scoring, show=show
    )

    # Fit and show the visualizer
    oz.fit(X, y)

    if show:
        oz.show()
    else:
        oz.finalize()

    # Return the visualizer object
    return oz


##########################################################################
## _RFECV
##########################################################################

class _RFECV(skRFECV):
    """
    A minor reimplementation of the :class:`~sklearn.feature_selection.RFECV` to store
    the cv scores so that we can compute the mean and standard deviation of the RFECV
    for visualization purposes.
    """

    def fit(self, X, y, groups=None):
        """
        Fit the RFE model and automatically tune the number of selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the total number of features.
        y : array-like of shape (n_samples,)
            Target values (integers for classification, real numbers for
            regression).
        groups : array-like of shape (n_samples,) or None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        """
        X, y = check_X_y(X, y, "csr", ensure_min_features=2,
                         force_all_finite=False)

        # Initialization
        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise YellowbrickValueError("step must be >0")

        # Build an RFE object, which will evaluate and score each possible
        # feature count, down to self.min_features_to_select
        rfe = RFE(estimator=self.estimator,
                  n_features_to_select=self.min_features_to_select,
                  step=self.step, verbose=self.verbose)

        # Determine the number of subsets of features by fitting across
        # the train folds and choosing the "features_to_select" parameter
        # that gives the least averaged error across all folds.

        # Note that joblib raises a non-picklable error for bound methods
        # even if n_jobs is set to 1 with the default multiprocessing
        # backend.
        # This branching is done so that to
        # make sure that user code that sets n_jobs to 1
        # and provides bound methods as scorers is not broken with the
        # addition of n_jobs parameter in version 0.18.

        if effective_n_jobs(self.n_jobs) == 1:
            parallel, func = list, _rfe_single_fit
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            func = delayed(_rfe_single_fit)

        scores = parallel(
            func(rfe, self.estimator, X, y, train, test, scorer)
            for train, test in cv.split(X, y, groups))

        # THIS IS THE NEW ADDITION
        self.cv_scores_ = np.asarray(scores)

        scores = np.sum(scores, axis=0)
        scores_rev = scores[::-1]
        argmax_idx = len(scores) - np.argmax(scores_rev) - 1
        n_features_to_select = max(
            n_features - (argmax_idx * step),
            self.min_features_to_select)

        # Re-execute an elimination with best_k over the whole set
        rfe = RFE(estimator=self.estimator,
                  n_features_to_select=n_features_to_select, step=self.step,
                  verbose=self.verbose)

        rfe.fit(X, y)

        # Set final attributes
        self.support_ = rfe.support_
        self.n_features_ = rfe.n_features_
        self.ranking_ = rfe.ranking_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self.transform(X), y)

        # Fixing a normalization error, n is equal to get_n_splits(X, y) - 1
        # here, the scores are normalized by get_n_splits(X, y)
        self.grid_scores_ = scores[::-1] / cv.get_n_splits(X, y, groups)
        return self
