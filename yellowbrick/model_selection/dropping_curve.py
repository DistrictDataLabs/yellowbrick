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
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest


# Default ticks for the model selection curve, relative number of features
DEFAULT_FEATURE_SIZES = np.linspace(0.1, 1.0, 5)


##########################################################################
# DroppingCurve visualizer
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
    feature_sizes_ : array,     shape = (n_unique_ticks,), dtype int
        Numbers of features that have been used to generate the
        dropping curve. Note that the number of ticks might be less
        than n_ticks because duplicate entries will be removed.

    train_scores_ : array, shape (n_ticks, n_cv_folds)
        Scores on training sets.

    train_scores_mean_ : array, shape (n_ticks,)
        Mean training data scores for each training split

    train_scores_std_ : array, shape (n_ticks,)
        Standard deviation of training data scores for each training split

    valid_scores_ : array, shape (n_ticks, n_cv_folds)
        Scores on validation set.

    valid_scores_mean_ : array, shape (n_ticks,)
        Mean scores for each validation split

    valid_scores_std_ : array, shape (n_ticks,)
        Standard deviation of  scores for each validation split

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
        ax=None,
        feature_sizes=DEFAULT_FEATURE_SIZES,
        groups=None,
        logx=False,
        cv=None,
        scoring=None,
        n_jobs=None,
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
        self.logx = logx
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fits the feature dropping curve with the wrapped model to the specified data.
        Draws training and cross-validation score curves and saves the scores to the
        estimator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        """
        # Get feature_sizes in whole numbers
        n_features = X.shape[-1]
        if np.issubdtype(self.feature_sizes.dtype, np.integer):
            if (self.feature_sizes <= 0).all() or (self.feature_sizes >= n_features).all():
                raise YellowbrickValueError('Expected feature sizes in [0, n_features]')
            self.feature_sizes_ = self.feature_sizes
        else:
            if (self.feature_sizes <= 0.0).all() or (self.feature_sizes >= 1.0).all():
                raise YellowbrickValueError('Expected feature ratio in [0,1]')
            self.feature_sizes_ = np.ceil(n_features * self.feature_sizes).astype(int)

        # The easiest way to prepend a random-dropout layer is to use
        # SelectKBest with a random scoring function.
        feature_dropping_pipeline = make_pipeline(
            SelectKBest(
                score_func=lambda X,y: np.random.default_rng(self.random_state).standard_normal(size=X.shape[-1])
                ),
            self.estimator,
        )

        # arguments to pass to sk_validation_curve
        skvc_kwargs = {
            key: self.get_params()[key]
            for key in (
                "groups",
                "cv",
                "scoring",
                "n_jobs",
                "pre_dispatch",
            )
        }

        self.train_scores_, self.valid_scores_ = sk_validation_curve(
            feature_dropping_pipeline,
            X,
            y,
            param_name="selectkbest__k",
            param_range=self.feature_sizes_,
            **skvc_kwargs
        )

        # compute the mean and standard deviation of the training data
        self.train_scores_mean_ = np.mean(self.train_scores_, axis=1)
        self.train_scores_std_ = np.std(self.train_scores_, axis=1)

        # compute the mean and standard deviation of the validation data
        self.valid_scores_mean_ = np.mean(self.valid_scores_, axis=1)
        self.valid_scores_std_ = np.std(self.valid_scores_, axis=1)

        # draw the curves on the current axes
        self.draw()
        return self

    def draw(self, **kwargs):
        """
        Renders the training and validation learning curves.
        """
        # Specify the curves to draw and their labels
        labels = ("Training Score", "Cross Validation Score")
        curves = (
            (self.train_scores_mean_, self.train_scores_std_),
            (self.valid_scores_mean_, self.valid_scores_std_),
        )

        # Get the colors for the train and test curves
        colors = resolve_colors(n_colors=2)

        # Plot the fill betweens first so they are behind the curves.
        for idx, (mean, std) in enumerate(curves):
            # Plot one standard deviation above and below the mean
            self.ax.fill_between(
                self.feature_sizes_, mean - std, mean + std, alpha=0.25, color=colors[idx]
            )

        # Plot the mean curves so they are in front of the variance fill
        for idx, (mean, _) in enumerate(curves):
            self.ax.plot(
                self.feature_sizes_, mean, "o-", color=colors[idx], label=labels[idx]
            )

        if self.logx:
            self.ax.set_xscale("log")

        return self.ax

    def finalize(self, **kwargs):
        """
        Add the title, legend, and other visual final touches to the plot.
        """
        # Set the title of the figure
        self.set_title("Random-feature dropping curve for {}".format(self.name))

        # Add the legend
        self.ax.legend(frameon=True, loc="best")

        # Set the axis labels
        self.ax.set_xlabel("number of features")
        self.ax.set_ylabel("score")


##########################################################################
# Quick Method
##########################################################################


def dropping_curve(
    estimator,
    X,
    y,
    feature_sizes=DEFAULT_FEATURE_SIZES,
    groups=None,
    ax=None,
    logx=False,
    cv=None,
    scoring=None,
    n_jobs=None,
    pre_dispatch='all',
    random_state=None,
    show=True,
    **kwargs
) -> DroppingCurve:
    """
    Displays a random-feature dropping curve, comparing feature size to training
    and cross validation scores. The dropping curve aims to show how a model
    improves with more information.

    This helper function wraps the DroppingCurve class for one-off analysis.

    Parameters
    ----------
    estimator : a scikit-learn estimator
        An object that implements ``fit`` and ``predict``, can be a
        classifier, regressor, or clusterer so long as there is also a valid
        associated scoring metric.

        Note that the object is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Input vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

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

    Returns
    -------
    dc : DroppingCurve
        Returns the fitted visualizer.
    """
    dc = DroppingCurve(
        estimator,
        feature_sizes=feature_sizes,
        groups=groups,
        ax=ax,
        logx=logx,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        pre_dispatch=pre_dispatch,
        random_state=random_state,
        **kwargs
    )

    # Fit and show the visualizer
    dc.fit(X, y)
    if show:
        dc.show()
    else:
        dc.finalize()
    return dc