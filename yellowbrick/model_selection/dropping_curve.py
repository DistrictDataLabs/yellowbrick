# yellowbrick.model_selection.dropping_curve
# Implements a feature dropping curve visualization for model selection.
#
# Author:   Charles Guan
# Created:  Wed Dec 8 15:03:00 2021 -0800

"""
Implements a random-input-dropout curve visualization for model selection.
Another common name: neuron dropping curve (NDC), in neural decoding research
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from yellowbrick.base import ModelVisualizer
from yellowbrick.style import resolve_colors
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.model_selection import validation_curve as sk_validation_curve


# Default ticks for the model selection curve, relative number of features
DEFAULT_FEATURE_SIZES = np.linspace(0.1, 1.0, 5)


##########################################################################
# ValidationCurve visualizer
##########################################################################


class DroppingCurve(ModelVisualizer):
    """
    Selects random subsets of features and estimates the training and
    crossvalidation performance. Subset sizes are swept to visualize a
    feature-dropping curve.

    The visualization plots the score relative to each subset and shows
    the number of (randomly selected) features needed to achieve a score.
    The curve is often shaped like log(1+x). For example, see:
    https://www.frontiersin.org/articles/10.3389/fnsys.2014.00102/full

    Parameters
    ----------
    estimator : a scikit-learn estimator
        An object that implements ``fit`` and ``predict``, can be a
        classifier, regressor, or clusterer so long as there is also a valid
        associated scoring metric.

        Note that the object is cloned for each validation.

    feature_sizes: array-like, shape (n_values,)
        default: ``np.linspace(0.1,1.0,5)``

        Relative or absolute numbers of input features that will be used to
        generate the learning curve. If the dtype is float, it is regarded as
        a fraction of the maximum number of features, otherwise it is
        interpreted as absolute numbers of features.

    groups : array-like, with shape (n_samples,)
        Optional group labels for the samples used while splitting the dataset
        into train/test sets.

    ax : matplotlib.Axes object, optional
        The axes object to plot the figure on.

    logx : boolean, optional
        If True, plots the x-axis with a logarithmic scale.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        see the scikit-learn
        `cross-validation guide <https://bit.ly/2MMQAI7>`_
        for more information on the possible strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string or scorer callable object / function with signature
        ``scorer(estimator, X, y)``. See scikit-learn model evaluation
        documentation for names of possible metrics.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    pre_dispatch : integer or string, optional
        Number of predispatched jobs for parallel execution (default is
        all). The option can reduce the allocated memory. The string can
        be an expression like '2*n_jobs'.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used to generate feature subsets.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    feature_sizes_ : array, shape = (n_unique_ticks,), dtype int
        Numbers of features that have been used to generate the
        dropping curve. Note that the number of ticks might be less
        than n_ticks because duplicate entries will be removed.

    train_scores_ : array, shape (n_ticks, n_cv_folds)
        Scores on training sets.

    train_scores_mean_ : array, shape (n_ticks,)
        Mean training data scores for each training split

    train_scores_std_ : array, shape (n_ticks,)
        Standard deviation of training data scores for each training split

    test_scores_ : array, shape (n_ticks, n_cv_folds)
        Scores on test set.

    test_scores_mean_ : array, shape (n_ticks,)
        Mean test data scores for each test split

    test_scores_std_ : array, shape (n_ticks,)
        Standard deviation of test data scores for each test split

    Examples
    --------

    >>> from yellowbrick.model_selection import DroppingCurve
    >>> from sklearn.naive_bayes import GaussianNB
    >>> model = DroppingCurve(GaussianNB())
    >>> model.fit(X, y)
    >>> model.show()

    Notes
    -----
    This visualizer is based on sklearn.model_selection.validation_curve
    """

    def __init__(
        self,
        estimator,
        feature_sizes=DEFAULT_FEATURE_SIZES,
        groups=None,
        ax=None,
        logx=False,
        cv=None,
        scoring=None,
        n_jobs=1,
        pre_dispatch='all',
        random_state=None,
        **kwargs
    ):

        # Initialize the model visualizer
        super(DroppingCurve, self).__init__(estimator, ax=ax, **kwargs)

        # Validate the feature sizes
        feature_sizes = np.asarray(feature_sizes)
        if feature_sizes.ndim != 1:
            raise YellowbrickValueError(
                "must specify 1-D array of feature sizes, '{}' is not valid".format(
                    repr(feature_sizes)
                )
            )

        # Set the metric parameters to be used later
        self.feature_sizes = feature_sizes
        self.groups = groups
        self.logx = self.logx
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.rng = np.random.default_rng(random_state)

