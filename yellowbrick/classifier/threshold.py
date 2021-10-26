# yellowbrick.classifier.threshold
# DiscriminationThreshold visualizer for probabilistic classifiers.
#
# Author:  Nathan Danielsen
# Author:  Benjamin Bengfort
# Created: Wed April 26 20:17:29 2017 -0700
#
# Copyright (C) 2017 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: threshold.py [] nathan.danielsen@gmail.com $

"""
DiscriminationThreshold visualizer for probabilistic classifiers.
"""

##########################################################################
## Imports
##########################################################################

import bisect
import numpy as np

from scipy.stats import mstats
from collections import defaultdict

from sklearn.base import clone
from sklearn.utils import indexable
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_recall_curve
from sklearn.utils.multiclass import type_of_target

try:
    # See #1137: this allows compatibility for scikit-learn >= 0.24
    from sklearn.utils import safe_indexing as _safe_indexing
except ImportError:
    from sklearn.utils import _safe_indexing

from yellowbrick.base import ModelVisualizer
from yellowbrick.style.colors import resolve_colors
from yellowbrick.utils import is_classifier, is_monotonic, is_probabilistic
from yellowbrick.exceptions import YellowbrickTypeError, YellowbrickValueError


# Quantiles for lower bound, curve, and upper bound
QUANTILES_MEDIAN_80 = np.array([0.1, 0.5, 0.9])

# List of threshold metrics
METRICS = ["precision", "recall", "fscore", "queue_rate"]


##########################################################################
# Discrimination Thresholds Visualization
##########################################################################


class DiscriminationThreshold(ModelVisualizer):
    """
    Visualizes how precision, recall, f1 score, and queue rate change as the
    discrimination threshold increases. For probabilistic, binary classifiers,
    the discrimination threshold is the probability at which you choose the
    positive class over the negative. Generally this is set to 50%, but
    adjusting the discrimination threshold will adjust sensitivity to false
    positives which is described by the inverse relationship of precision and
    recall with respect to the threshold.

    The visualizer also accounts for variability in the model by running
    multiple trials with different train and test splits of the data. The
    variability is visualized using a band such that the curve is drawn as the
    median score of each trial and the band is from the 10th to 90th
    percentile.

    The visualizer is intended to help users determine an appropriate
    threshold for decision making (e.g. at what threshold do we have a human
    review the data), given a tolerance for precision and recall or limiting
    the number of records to check (the queue rate).

    Parameters
    ----------
    estimator : estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If not specified the current axes will be
        used (or generated if required).

    n_trials : integer, default: 50
        Number of times to shuffle and split the dataset to account for noise
        in the threshold metrics curves. Note if cv provides > 1 splits,
        the number of trials will be n_trials * cv.get_n_splits()

    cv : float or cross-validation generator, default: 0.1
        Determines the splitting strategy for each trial. Possible inputs are:

        - float, to specify the percent of the test split
        - object to be used as cross-validation generator

        This attribute is meant to give flexibility with stratified splitting
        but if a splitter is provided, it should only return one split and
        have shuffle set to True.

    fbeta : float, 1.0 by default
        The strength of recall versus precision in the F-score.

    argmax : str or None, default: 'fscore'
        Annotate the threshold maximized by the supplied metric (see exclude
        for the possible metrics to use). If None or passed to exclude,
        will not annotate the graph.

    exclude : str or list, optional
        Specify metrics to omit from the graph, can include:

        - ``"precision"``
        - ``"recall"``
        - ``"queue_rate"``
        - ``"fscore"``

        Excluded metrics will not be displayed in the graph, nor will they
        be available in ``thresholds_``; however, they will be computed on fit.

    quantiles : sequence, default: np.array([0.1, 0.5, 0.9])
        Specify the quantiles to view model variability across a number of
        trials. Must be monotonic and have three elements such that the first
        element is the lower bound, the second is the drawn curve, and the
        third is the upper bound. By default the curve is drawn at the median,
        and the bounds from the 10th percentile to the 90th percentile.

    random_state : int, optional
        Used to seed the random state for shuffling the data while composing
        different train and test splits. If supplied, the random state is
        incremented in a deterministic fashion for each split.

        Note that if a splitter is provided, it's random state will also be
        updated with this random state, even if it was previously set.

    is_fitted : bool or str, default="auto"
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If "auto" (default), a helper method will check if the estimator
        is fitted before fitting it again.

    force_model : bool, default: False
        Do not check to ensure that the underlying estimator is a classifier. This
        will prevent an exception when the visualizer is initialized but may result
        in unexpected or unintended behavior.

    kwargs : dict
        Keyword arguments passed to the visualizer base classes.

    Attributes
    ----------
    thresholds_ : array
        The uniform thresholds identified by each of the trial runs.

    cv_scores_ : dict of arrays of ``len(thresholds_)``
        The values for all included metrics including the upper and lower
        bounds of the metrics defined by quantiles.

    Notes
    -----
    The term "discrimination threshold" is rare in the literature. Here, we
    use it to mean the probability at which the positive class is selected
    over the negative class in binary classification.

    Classification models must implement either a ``decision_function`` or
    ``predict_proba`` method in order to be used with this class. A
    ``YellowbrickTypeError`` is raised otherwise.

    .. caution:: This method only works for binary, probabilistic classifiers.

    .. seealso::
        For a thorough explanation of discrimination thresholds, see:
        `Visualizing Machine Learning Thresholds to Make Better Business
        Decisions
        <http://blog.insightdatalabs.com/visualizing-classifier-thresholds/>`_
        by Insight Data.
    """

    def __init__(
        self,
        estimator,
        ax=None,
        n_trials=50,
        cv=0.1,
        fbeta=1.0,
        argmax="fscore",
        exclude=None,
        quantiles=QUANTILES_MEDIAN_80,
        random_state=None,
        is_fitted="auto",
        force_model=False,
        **kwargs
    ):

        # Perform some quick type checking to help users avoid error.
        if not force_model and (
            not is_classifier(estimator) or not is_probabilistic(estimator)
        ):
            raise YellowbrickTypeError(
                "{} requires a probabilistic binary classifier".format(
                    self.__class__.__name__
                )
            )

        # Check the various inputs
        self._check_quantiles(quantiles)
        self._check_cv(cv)
        self._check_exclude(exclude)
        self._check_argmax(argmax, exclude)

        # Initialize the ModelVisualizer
        super(DiscriminationThreshold, self).__init__(
            estimator, ax=ax, is_fitted=is_fitted, **kwargs
        )

        # Set params
        self.n_trials = n_trials
        self.cv = cv
        self.fbeta = fbeta
        self.argmax = argmax
        self.exclude = exclude
        self.quantiles = quantiles
        self.random_state = random_state

    def fit(self, X, y, **kwargs):
        """
        Fit is the entry point for the visualizer. Given instances described
        by X and binary classes described in the target y, fit performs n
        trials by shuffling and splitting the dataset then computing the
        precision, recall, f1, and queue rate scores for each trial. The
        scores are aggregated by the quantiles expressed then drawn.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values. The target y must
            be a binary classification target.

        kwargs: dict
            keyword arguments passed to Scikit-Learn API.

        Returns
        -------
        self : instance
            Returns the instance of the visualizer

        raises: YellowbrickValueError
            If the target y is not a binary classification target.
        """
        # Check target before metrics raise crazy exceptions
        # TODO: Switch to using target type from utils.target
        if type_of_target(y) != "binary":
            raise YellowbrickValueError("multiclass format is not supported")

        # Make arrays indexable for cross validation
        X, y = indexable(X, y)

        # TODO: parallelize trials with joblib (using sklearn utility)
        # NOTE: parallelization with matplotlib is tricky at best!
        trials = [
            metric
            for idx in range(self.n_trials)
            for metric in self._split_fit_score_trial(X, y, idx)
        ]

        # Compute maximum number of uniform thresholds across all trials
        n_thresholds = np.array([len(t["thresholds"]) for t in trials]).min()
        self.thresholds_ = np.linspace(0.0, 1.0, num=n_thresholds)

        # Filter metrics and collect values for uniform thresholds
        metrics = frozenset(METRICS) - self._check_exclude(self.exclude)
        uniform_metrics = defaultdict(list)

        for trial in trials:
            rows = defaultdict(list)
            for t in self.thresholds_:
                idx = bisect.bisect_left(trial["thresholds"], t)
                for metric in metrics:
                    rows[metric].append(trial[metric][idx])

            for metric, row in rows.items():
                uniform_metrics[metric].append(row)

        # Convert metrics to metric arrays
        uniform_metrics = {
            metric: np.array(values) for metric, values in uniform_metrics.items()
        }

        # Perform aggregation and store cv_scores_
        quantiles = self._check_quantiles(self.quantiles)
        self.cv_scores_ = {}

        for metric, values in uniform_metrics.items():
            # Compute the lower, median, and upper plots
            lower, median, upper = mstats.mquantiles(values, prob=quantiles, axis=0)

            # Store the aggregates in cv scores
            self.cv_scores_[metric] = median
            self.cv_scores_["{}_lower".format(metric)] = lower
            self.cv_scores_["{}_upper".format(metric)] = upper

        # TODO: fit the underlying estimator with the best decision threshold
        # Call super to ensure the underlying estimator is correctly fitted
        super(DiscriminationThreshold, self).fit(X, y)

        # Draw and always return self
        self.draw()
        return self

    def _split_fit_score_trial(self, X, y, idx=0):
        """
        Splits the dataset, fits a clone of the estimator, then scores it
        according to the required metrics.

        The index of the split is added to the random_state if the
        random_state is not None; this ensures that every split is shuffled
        differently but in a deterministic fashion for testing purposes.
        """
        random_state = self.random_state
        if random_state is not None:
            random_state += idx

        splitter = self._check_cv(self.cv, random_state)

        for train_index, test_index in splitter.split(X, y):
            # Safe indexing handles multiple types of inputs including
            # DataFrames and structured arrays - required for generic splits.
            X_train = _safe_indexing(X, train_index)
            y_train = _safe_indexing(y, train_index)
            X_test = _safe_indexing(X, test_index)
            y_test = _safe_indexing(y, test_index)

            model = clone(self.estimator)
            model.fit(X_train, y_train)

            if hasattr(model, "predict_proba"):
                # Get the probabilities for the positive class
                y_scores = model.predict_proba(X_test)[:, 1]
            else:
                # Use the decision function to get the scores
                y_scores = model.decision_function(X_test)

            # Compute the curve metrics and thresholds
            curve_metrics = precision_recall_curve(y_test, y_scores)
            precision, recall, thresholds = curve_metrics

            # Compute the F1 score from precision and recall
            # Don't need to warn for F, precision/recall would have warned
            with np.errstate(divide="ignore", invalid="ignore"):
                beta = self.fbeta ** 2
                f_score = (1 + beta) * precision * recall / (beta * precision + recall)

            # Ensure thresholds ends at 1
            thresholds = np.append(thresholds, 1)

            # Compute the queue rate
            queue_rate = np.array(
                [(y_scores >= threshold).mean() for threshold in thresholds]
            )

            yield {
                "thresholds": thresholds,
                "precision": precision,
                "recall": recall,
                "fscore": f_score,
                "queue_rate": queue_rate,
            }

    def draw(self):
        """
        Draws the cv scores as a line chart on the current axes.
        """
        # Set the colors from the supplied values or reasonable defaults
        color_values = resolve_colors(n_colors=4, colors=self.color)

        # Get the metric used to annotate the graph with its maximizing value
        argmax = self._check_argmax(self.argmax, self.exclude)

        for idx, metric in enumerate(METRICS):
            # Skip any excluded labels
            if metric not in self.cv_scores_:
                continue

            # Get the color ensuring every metric has a static color
            color = color_values[idx]

            # Make the label pretty
            if metric == "fscore":
                if self.fbeta == 1.0:
                    label = "$f_1$"
                else:
                    label = "$f_{{\beta={:0.1f}}}".format(self.fbeta)
            else:
                label = metric.replace("_", " ")

            # Draw the metric values
            self.ax.plot(
                self.thresholds_, self.cv_scores_[metric], color=color, label=label
            )

            # Draw the upper and lower bounds
            lower = self.cv_scores_["{}_lower".format(metric)]
            upper = self.cv_scores_["{}_upper".format(metric)]

            self.ax.fill_between(
                self.thresholds_, upper, lower, alpha=0.35, linewidth=0, color=color
            )

            # Annotate the graph with the maximizing value
            if argmax and argmax == metric:
                argmax = self.cv_scores_[metric].argmax()
                threshold = self.thresholds_[argmax]
                self.ax.axvline(
                    threshold,
                    ls="--",
                    c="k",
                    lw=1,
                    label="$t_{}={:0.2f}$".format(metric[0], threshold),
                )

        return self.ax

    def finalize(self, **kwargs):
        """
        Sets a title and axis labels on the visualizer and ensures that the
        axis limits are scaled to valid threshold values.

        Parameters
        ----------
        kwargs: generic keyword arguments.

        Notes
        -----
        Generally this method is called from show and not directly by the user.
        """
        super(DiscriminationThreshold, self).finalize(**kwargs)

        # Set the title of the threshold visualiztion
        self.set_title("Threshold Plot for {}".format(self.name))

        self.ax.legend(frameon=True, loc="best")
        self.ax.set_xlabel("discrimination threshold")
        self.ax.set_ylabel("score")
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)

    def _check_quantiles(self, val):
        """
        Validate the quantiles passed in. Returns the np array if valid.
        """
        if len(val) != 3 or not is_monotonic(val) or not np.all(val < 1):
            raise YellowbrickValueError(
                "quantiles must be a sequence of three "
                "monotonically increasing values less than 1"
            )
        return np.asarray(val)

    def _check_cv(self, val, random_state=None):
        """
        Validate the cv method passed in. Returns the split strategy if no
        validation exception is raised.
        """
        # Use default splitter in this case
        if val is None:
            val = 0.1

        if isinstance(val, float) and val <= 1.0:
            return ShuffleSplit(n_splits=1, test_size=val, random_state=random_state)

        if hasattr(val, "split") and hasattr(val, "get_n_splits"):
            if random_state is not None and hasattr(val, "random_state"):
                val.random_state = random_state
            return val

        raise YellowbrickValueError("'{}' is not a valid cv splitter".format(type(val)))

    def _check_exclude(self, val):
        """
        Validate the excluded metrics. Returns the set of excluded params.
        """
        if val is None:
            exclude = frozenset()
        elif isinstance(val, str):
            exclude = frozenset([val.lower()])
        else:
            exclude = frozenset(map(lambda s: s.lower(), val))

        if len(exclude - frozenset(METRICS)) > 0:
            raise YellowbrickValueError(
                "'{}' is not a valid metric to exclude".format(repr(val))
            )

        return exclude

    def _check_argmax(self, val, exclude=None):
        """
        Validate the argmax metric. Returns the metric used to annotate the graph.
        """
        if val is None:
            return None

        argmax = val.lower()

        if argmax not in METRICS:
            raise YellowbrickValueError(
                "'{}' is not a valid metric to use".format(repr(val))
            )

        exclude = self._check_exclude(exclude)
        if argmax in exclude:
            argmax = None

        return argmax


##########################################################################
# Quick Methods
##########################################################################

def discrimination_threshold(
    estimator,
    X,
    y,
    ax=None,
    n_trials=50,
    cv=0.1,
    fbeta=1.0,
    argmax="fscore",
    exclude=None,
    quantiles=QUANTILES_MEDIAN_80,
    random_state=None,
    is_fitted="auto",
    force_model=False,
    show=True,
    **kwargs
):
    """Discrimination Threshold

    Visualizes how precision, recall, f1 score, and queue rate change as the
    discrimination threshold increases. For probabilistic, binary classifiers,
    the discrimination threshold is the probability at which you choose the
    positive class over the negative. Generally this is set to 50%, but
    adjusting the discrimination threshold will adjust sensitivity to false
    positives which is described by the inverse relationship of precision and
    recall with respect to the threshold.

    .. seealso:: See DiscriminationThreshold for more details.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.

    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features

    y : ndarray or Series of length n
        An array or series of target or class values. The target y must
        be a binary classification target.

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If not specified the current axes will be
        used (or generated if required).

    n_trials : integer, default: 50
        Number of times to shuffle and split the dataset to account for noise
        in the threshold metrics curves. Note if cv provides > 1 splits,
        the number of trials will be n_trials * cv.get_n_splits()

    cv : float or cross-validation generator, default: 0.1
        Determines the splitting strategy for each trial. Possible inputs are:

        - float, to specify the percent of the test split
        - object to be used as cross-validation generator

        This attribute is meant to give flexibility with stratified splitting
        but if a splitter is provided, it should only return one split and
        have shuffle set to True.

    fbeta : float, 1.0 by default
        The strength of recall versus precision in the F-score.

    argmax : str or None, default: 'fscore'
        Annotate the threshold maximized by the supplied metric (see exclude
        for the possible metrics to use). If None or passed to exclude,
        will not annotate the graph.

    exclude : str or list, optional
        Specify metrics to omit from the graph, can include:

        - ``"precision"``
        - ``"recall"``
        - ``"queue_rate"``
        - ``"fscore"``

        Excluded metrics will not be displayed in the graph, nor will they
        be available in ``thresholds_``; however, they will be computed on fit.

    quantiles : sequence, default: np.array([0.1, 0.5, 0.9])
        Specify the quantiles to view model variability across a number of
        trials. Must be monotonic and have three elements such that the first
        element is the lower bound, the second is the drawn curve, and the
        third is the upper bound. By default the curve is drawn at the median,
        and the bounds from the 10th percentile to the 90th percentile.

    random_state : int, optional
        Used to seed the random state for shuffling the data while composing
        different train and test splits. If supplied, the random state is
        incremented in a deterministic fashion for each split.

        Note that if a splitter is provided, it's random state will also be
        updated with this random state, even if it was previously set.

    is_fitted : bool or str, default="auto"
        Specify if the wrapped estimator is already fitted. If False, the estimator
        will be fit when the visualizer is fit, otherwise, the estimator will not be
        modified. If "auto" (default), a helper method will check if the estimator
        is fitted before fitting it again.

    force_model : bool, default: False
        Do not check to ensure that the underlying estimator is a classifier. This
        will prevent an exception when the visualizer is initialized but may result
        in unexpected or unintended behavior.

    show : bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot
        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply
        calls ``finalize()``

    kwargs : dict
        Keyword arguments passed to the visualizer base classes.

    Notes
    -----
    The term "discrimination threshold" is rare in the literature. Here, we
    use it to mean the probability at which the positive class is selected
    over the negative class in binary classification.

    Classification models must implement either a ``decision_function`` or
    ``predict_proba`` method in order to be used with this class. A
    ``YellowbrickTypeError`` is raised otherwise.

    .. seealso::
        For a thorough explanation of discrimination thresholds, see:
        `Visualizing Machine Learning Thresholds to Make Better Business
        Decisions
        <http://blog.insightdatalabs.com/visualizing-classifier-thresholds/>`_
        by Insight Data.

    Examples
    --------
    >>> from yellowbrick.classifier.threshold import discrimination_threshold
    >>> from sklearn.linear_model import LogisticRegression
    >>> from yellowbrick.datasets import load_occupancy
    >>> X, y = load_occupancy()
    >>> model = LogisticRegression(multi_class="auto", solver="liblinear")
    >>> discrimination_threshold(model, X, y)

    Returns
    -------
    viz : DiscriminationThreshold
        Returns the fitted and finalized visualizer object.
    """
    # Instantiate the visualizer
    visualizer = DiscriminationThreshold(
        estimator,
        ax=ax,
        n_trials=n_trials,
        cv=cv,
        fbeta=fbeta,
        argmax=argmax,
        exclude=exclude,
        quantiles=quantiles,
        random_state=random_state,
        is_fitted=is_fitted,
        force_model=force_model,
        **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y)

    if show:
        visualizer.show()
    else:
        visualizer.finalize()

    # Return the visualizer
    return visualizer
