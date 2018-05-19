# yellowbrick.classifier.threshold
# DiscriminationThreshold visualizer for probabilistic classifiers.
#
# Author:  Nathan Danielsen <ndanielsen@gmail.com>
# Author:  Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created: Wed April 26 20:17:29 2017 -0700
#
# ID: threshold.py [] nathan.danielsen@gmail.com $

"""
DiscriminationThreshold visualizer for probabilistic classifiers.
"""

##########################################################################
## Imports
##########################################################################

import six
import bisect
import numpy as np

from scipy.stats import mstats
from collections import defaultdict

from yellowbrick.base import ModelVisualizer
from yellowbrick.style.colors import resolve_colors
from yellowbrick.utils import is_classifier, is_probabilistic, is_monotonic
from yellowbrick.exceptions import YellowbrickTypeError, YellowbrickValueError

from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit
from sklearn.utils.deprecation import deprecated
from sklearn.metrics import precision_recall_curve
from sklearn.utils import indexable, safe_indexing
from sklearn.utils.multiclass import type_of_target


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

    .. caution:: This method only works for binary, probabilistic classifiers.

    Parameters
    ----------
    model : Classification Estimator
        A binary classification estimator that implements ``predict_proba`` or
        ``decision_function`` methods. Will raise ``TypeError`` if the model
        cannot be used with the visualizer.

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

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

    argmax : str, default: 'fscore'
        Annotate the threshold maximized by the supplied metric (see exclude
        for the possible metrics to use). If None, will not annotate the
        graph.

    exclude : str or list, optional
        Specify metrics to omit from the graph, can include:

        - ``"precision"``
        - ``"recall"``
        - ``"queue_rate"``
        - ``"fscore"``

        All metrics not excluded will be displayed in the graph, nor will they
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

    kwargs : dict
        Keyword arguments that are passed to the base visualizer class.

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

    .. seealso::
        For a thorough explanation of discrimination thresholds, see:
        `Visualizing Machine Learning Thresholds to Make Better Business
        Decisions
        <http://blog.insightdatalabs.com/visualizing-classifier-thresholds/>`_
        by Insight Data.
    """

    def __init__(self, model, ax=None, n_trials=50, cv=0.1, fbeta=1.0,
                 argmax='fscore', exclude=None, quantiles=QUANTILES_MEDIAN_80,
                 random_state=None, **kwargs):

        # Perform some quick type checking to help users avoid error.
        if not is_classifier(model) or not is_probabilistic(model):
            raise YellowbrickTypeError(
                "{} requires a probabilistic binary classifier".format(
                self.__class__.__name__
            ))

        # Check the various inputs
        self._check_quantiles(quantiles)
        self._check_cv(cv)
        self._check_exclude(exclude)

        # Initialize the ModelVisualizer
        super(DiscriminationThreshold, self).__init__(model, ax=ax, **kwargs)

        # Set params
        self.set_params(
            n_trials=n_trials, cv=cv, fbeta=fbeta, argmax=argmax,
            exclude=exclude, quantiles=quantiles, random_state=random_state,
        )


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
        if type_of_target(y) != 'binary':
            raise YellowbrickValueError("multiclass format is not supported")

        # Make arrays indexable for cross validation
        X, y = indexable(X, y)

        # TODO: parallelize trials with joblib (using sklearn utility)
        # NOTE: parallelization with matplotlib is tricy at best!
        trials = [
            metric
            for idx in range(self.n_trials)
            for metric in self._split_fit_score_trial(X, y, idx)
        ]

        # Compute maximum number of uniform thresholds across all trials
        n_thresholds = np.array([len(t['thresholds']) for t in trials]).min()
        self.thresholds_ = np.linspace(0.0, 1.0, num=n_thresholds)

        # Filter metrics and collect values for uniform thresholds
        metrics = frozenset(METRICS) - self._check_exclude(self.exclude)
        uniform_metrics = defaultdict(list)

        for trial in trials:
            rows = defaultdict(list)
            for t in self.thresholds_:
                idx = bisect.bisect_left(trial['thresholds'], t)
                for metric in metrics:
                    rows[metric].append(trial[metric][idx])

            for metric, row in rows.items():
                uniform_metrics[metric].append(row)

        # Convert metrics to metric arrays
        uniform_metrics = {
            metric: np.array(values)
            for metric, values in uniform_metrics.items()
        }

        # Perform aggregation and store cv_scores_
        quantiles = self._check_quantiles(self.quantiles)
        self.cv_scores_ = {}

        for metric, values in uniform_metrics.items():
            # Compute the lower, median, and upper plots
            lower, median, upper = mstats.mquantiles(
                values, prob=quantiles, axis=0
            )

            # Store the aggregates in cv scores
            self.cv_scores_[metric] = median
            self.cv_scores_["{}_lower".format(metric)] = lower
            self.cv_scores_["{}_upper".format(metric)] = upper

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
            X_train = safe_indexing(X, train_index)
            y_train = safe_indexing(y, train_index)
            X_test = safe_indexing(X, test_index)
            y_test = safe_indexing(y, test_index)

            model = clone(self.estimator)
            model.fit(X_train, y_train)

            if hasattr(model, "predict_proba"):
                # Get the probabilities for the positive class
                y_scores = model.predict_proba(X_test)[:,1]
            else:
                # Use the decision function to get the scores
                y_scores = model.decision_function(X_test)

            # Compute the curve metrics and thresholds
            curve_metrics = precision_recall_curve(y_test, y_scores)
            precision, recall, thresholds = curve_metrics

            # Compute the F1 score from precision and recall
            # Don't need to warn for F, precision/recall would have warned
            with np.errstate(divide='ignore', invalid='ignore'):
                beta = self.fbeta ** 2
                f_score = ((1 + beta) * precision * recall /
                   (beta * precision + recall))

            # Ensure thresholds ends at 1
            thresholds = np.append(thresholds, 1)

            # Compute the queue rate
            queue_rate = np.array([
                (y_scores >= threshold).mean()
                for threshold in thresholds
            ])

            yield {
                'thresholds': thresholds,
                'precision': precision,
                'recall': recall,
                'fscore': f_score,
                'queue_rate': queue_rate
            }

    def draw(self):
        """
        Draws the cv scores as a line chart on the current axes.
        """
        # Set the colors from the supplied values or reasonable defaults
        color_values = resolve_colors(n_colors=4, colors=self.color)

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
                self.thresholds_, self.cv_scores_[metric],
                color=color, label=label
            )

            # Draw the upper and lower bounds
            lower = self.cv_scores_["{}_lower".format(metric)]
            upper = self.cv_scores_["{}_upper".format(metric)]

            self.ax.fill_between(
                self.thresholds_, upper, lower,
                alpha=0.35, linewidth=0, color=color
            )

            # Annotate the graph with the maximizing value
            if self.argmax.lower() == metric:
                argmax = self.cv_scores_[metric].argmax()
                threshold = self.thresholds_[argmax]
                self.ax.axvline(
                    threshold, ls='--', c='k', lw=1,
                    label="$t_{}={:0.2f}$".format(metric[0], threshold)
                )

        return self.ax

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.
        The user calls poof and poof calls finalize.

        Parameters
        ----------
        kwargs: generic keyword arguments.
        """
        super(DiscriminationThreshold, self).finalize(**kwargs)

        # Set the title of the threshold visualiztion
        self.set_title("Threshold Plot for {}".format(self.name))

        self.ax.legend(frameon=True, loc='best')
        self.ax.set_xlabel('discrimination threshold')
        self.ax.set_ylabel('score')
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
        if val is None: val = 0.1

        if isinstance(val, float) and val <= 1.0:
            return ShuffleSplit(
                n_splits=1, test_size=val, random_state=random_state
            )

        if hasattr(val, "split") and hasattr(val, "get_n_splits"):
            if random_state is not None and hasattr(val, "random_state"):
                val.random_state = random_state
            return val

        raise YellowbrickValueError(
            "'{}' is not a valid cv splitter".format(type(val))
        )

    def _check_exclude(self, val):
        """
        Validate the excluded metrics. Returns the set of excluded params.
        """
        if val is None:
            exclude = frozenset()
        elif isinstance(val, six.string_types):
            exclude = frozenset([val.lower()])
        else:
            exclude = frozenset(map(lambda s: s.lower(), val))

        if len(exclude - frozenset(METRICS)) > 0:
            raise YellowbrickValueError(
                "'{}' is not a valid metric to exclude".format(repr(val))
            )

        return exclude


##########################################################################
# Quick Methods
##########################################################################

def discrimination_threshold(model, X, y, ax=None, n_trials=50, cv=0.1,
                             fbeta=1.0, argmax='fscore', exclude=None,
                             quantiles=QUANTILES_MEDIAN_80, random_state=None,
                             **kwargs):
    """Quick method for DiscriminationThreshold.

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
    model : Classification Estimator
        A binary classification estimator that implements ``predict_proba`` or
        ``decision_function`` methods. Will raise ``TypeError`` if the model
        cannot be used with the visualizer.

    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features

    y : ndarray or Series of length n
        An array or series of target or class values. The target y must
        be a binary classification target.

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

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

    argmax : str, default: 'fscore'
        Annotate the threshold maximized by the supplied metric (see exclude
        for the possible metrics to use). If None, will not annotate the
        graph.

    exclude : str or list, optional
        Specify metrics to omit from the graph, can include:

        - ``"precision"``
        - ``"recall"``
        - ``"queue_rate"``
        - ``"fscore"``

        All metrics not excluded will be displayed in the graph, nor will they
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

    kwargs : dict
        Keyword arguments that are passed to the base visualizer class.

    Returns
    -------
    ax : matplotlib axes
        Returns the axes that the parallel coordinates were drawn on.
    """
    # Instantiate the visualizer
    visualizer = DiscriminationThreshold(
        model, ax=ax, n_trials=n_trials, cv=cv,  fbeta=fbeta, argmax=argmax,
        exclude=exclude,  quantiles=quantiles, random_state=random_state,
        **kwargs
    )

    # Fit and transform the visualizer (calls draw)
    visualizer.fit(X, y)
    visualizer.poof()

    # Return the axes object on the visualizer
    return visualizer.ax


##########################################################################
## Aliases (Deprecated)
##########################################################################

@deprecated("alias for DiscriminationThreshold will be removed in v0.8")
class ThresholdVisualizer(DiscriminationThreshold):
    pass


@deprecated("alias for DiscriminationThreshold will be removed in v0.8")
class ThreshViz(DiscriminationThreshold):
    pass
