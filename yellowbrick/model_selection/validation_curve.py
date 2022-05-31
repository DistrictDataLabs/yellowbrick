# yellowbrick.model_selection.validation_curve
# Implements a visual validation curve for a hyperparameter.
#
# Author:  Benjamin Bengfort
# Created: Sat Mar 31 06:27:28 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: validation_curve.py [c5355ee] benjamin@bengfort.com $

"""
Implements a visual validation curve for a hyperparameter.
"""

##########################################################################
# Imports
##########################################################################

import numpy as np

from yellowbrick.base import ModelVisualizer
from yellowbrick.style import resolve_colors
from yellowbrick.exceptions import YellowbrickValueError

from sklearn.model_selection import validation_curve as sk_validation_curve


##########################################################################
# ValidationCurve visualizer
##########################################################################


class ValidationCurve(ModelVisualizer):
    """
    Visualizes the validation curve for both test and training data for a
    range of values for a single hyperparameter of the model. Adjusting the
    value of a hyperparameter adjusts the complexity of a model. Less complex
    models suffer from increased error due to bias, while more complex models
    suffer from increased error due to variance. By inspecting the training
    and cross-validated test score error, it is possible to estimate a good
    value for a hyperparameter that balances the bias/variance trade-off.

    The visualizer evaluates cross-validated training and test scores for the
    different hyperparameters supplied. The curve is plotted so that the
    x-axis is the value of the hyperparameter and the y-axis is the model
    score. This is similar to a grid search with a single hyperparameter.

    The cross-validation generator splits the dataset k times, and scores are
    averaged over all k runs for the training and test subsets. The curve
    plots the mean score, and the filled in area suggests the variability of
    cross-validation by plotting one standard deviation above and below the
    mean for each split.

    Parameters
    ----------
    estimator : a scikit-learn estimator
        An object that implements ``fit`` and ``predict``, can be a
        classifier, regressor, or clusterer so long as there is also a valid
        associated scoring metric.

        Note that the object is cloned for each validation.

    param_name : string
        Name of the parameter that will be varied.

    param_range : array-like, shape (n_values,)
        The values of the parameter that will be evaluated.

    ax : matplotlib.Axes object, optional
        The axes object to plot the figure on.

    logx : boolean, optional
        If True, plots the x-axis with a logarithmic scale.

    groups : array-like, with shape (n_samples,)
        Optional group labels for the samples used while splitting the dataset
        into train/test sets.

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

    markers : string, default: '-d'
        Matplotlib style markers for points on the plot points
        Options: '-,', '-+', '-o', '-*', '-v', '-h', '-d'

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
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

    >>> import numpy as np
    >>> from yellowbrick.model_selection import ValidationCurve
    >>> from sklearn.svm import SVC
    >>> pr = np.logspace(-6,-1,5)
    >>> model = ValidationCurve(SVC(), param_name="gamma", param_range=pr)
    >>> model.fit(X, y)
    >>> model.show()

    Notes
    -----
    This visualizer is essentially a wrapper for the
    ``sklearn.model_selection.learning_curve utility``, discussed in the
    `validation curves <https://bit.ly/2KlumeB>`__
    documentation.

    .. seealso:: The documentation for the
        `learning_curve <https://bit.ly/2Yz9sBB>`__
        function, which this visualizer wraps.
    """

    def __init__(
        self,
        estimator,
        param_name,
        param_range,
        ax=None,
        logx=False,
        groups=None,
        cv=None,
        scoring=None,
        n_jobs=1,
        pre_dispatch="all",
        markers='-d',
        **kwargs
    ):

        # Initialize the model visualizer
        super(ValidationCurve, self).__init__(estimator, ax=ax, **kwargs)

        # Validate the param_range
        param_range = np.asarray(param_range)
        if param_range.ndim != 1:
            raise YellowbrickValueError(
                "must specify array of param values, '{}' is not valid".format(
                    repr(param_range)
                )
            )

        # Set the visual and validation curve parameters on the estimator
        self.param_name = param_name
        self.param_range = param_range
        self.logx = logx
        self.groups = groups
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.markers = markers

    def fit(self, X, y=None):
        """
        Fits the validation curve with the wrapped estimator and parameter
        array to the specified data. Draws training and test score curves and
        saves the scores to the visualizer.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        self : instance
            Returns the instance of the validation curve visualizer for use in
            pipelines and other sequential transformers.
        """
        # arguments to pass to sk_validation_curve
        skvc_kwargs = {
            key: self.get_params()[key]
            for key in (
                "param_name",
                "param_range",
                "groups",
                "cv",
                "scoring",
                "n_jobs",
                "pre_dispatch",
            )
        }

        # compute the validation curve and store scores
        curve = sk_validation_curve(self.estimator, X, y, **skvc_kwargs)
        self.train_scores_, self.test_scores_ = curve

        # compute the mean and standard deviation of the training data
        self.train_scores_mean_ = np.mean(self.train_scores_, axis=1)
        self.train_scores_std_ = np.std(self.train_scores_, axis=1)

        # compute the mean and standard deviation of the test data
        self.test_scores_mean_ = np.mean(self.test_scores_, axis=1)
        self.test_scores_std_ = np.std(self.test_scores_, axis=1)

        # draw the curves on the current axes
        self.draw()
        return self

    def draw(self, **kwargs):
        """
        Renders the training and test curves.
        """
        # Specify the curves to draw and their labels
        labels = ("Training Score", "Cross Validation Score")
        curves = (
            (self.train_scores_mean_, self.train_scores_std_),
            (self.test_scores_mean_, self.test_scores_std_),
        )

        # Get the colors for the train and test curves
        colors = resolve_colors(n_colors=2)

        # Plot the fill betweens first so they are behind the curves.
        for idx, (mean, std) in enumerate(curves):
            # Plot one standard deviation above and below the mean
            self.ax.fill_between(
                self.param_range, mean - std, mean + std, alpha=0.25, color=colors[idx]
            )

        # Plot the mean curves so they are in front of the variance fill
        for idx, (mean, _) in enumerate(curves):
            self.ax.plot(
                self.param_range, mean, self.markers, color=colors[idx], label=labels[idx]
            )

        if self.logx:
            self.ax.set_xscale("log")

        return self.ax

    def finalize(self, **kwargs):
        """
        Add the title, legend, and other visual final touches to the plot.
        """
        # Set the title of the figure
        self.set_title("Validation Curve for {}".format(self.name))

        # Add the legend
        self.ax.legend(frameon=True, loc="best")

        # Set the axis labels
        self.ax.set_xlabel(self.param_name)
        self.ax.set_ylabel("score")


##########################################################################
# Quick Method
##########################################################################


def validation_curve(
    estimator,
    X,
    y,
    param_name,
    param_range,
    ax=None,
    logx=False,
    groups=None,
    cv=None,
    scoring=None,
    n_jobs=1,
    pre_dispatch="all",
    show=True,
    markers='-d',
    **kwargs
):
    """
    Displays a validation curve for the specified param and values, plotting
    both the train and cross-validated test scores. The validation curve is a
    visual, single-parameter grid search used to tune a model to find the best
    balance between error due to bias and error due to variance.

    This helper function is a wrapper to use the ValidationCurve in a fast,
    visual analysis.

    Parameters
    ----------
    estimator : a scikit-learn estimator
        An object that implements ``fit`` and ``predict``, can be a
        classifier, regressor, or clusterer so long as there is also a valid
        associated scoring metric.

        Note that the object is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    param_name : string
        Name of the parameter that will be varied.

    param_range : array-like, shape (n_values,)
        The values of the parameter that will be evaluated.

    ax : matplotlib.Axes object, optional
        The axes object to plot the figure on.

    logx : boolean, optional
        If True, plots the x-axis with a logarithmic scale.

    groups : array-like, with shape (n_samples,)
        Optional group labels for the samples used while splitting the dataset
        into train/test sets.

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

    show: bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however
        you cannot call ``plt.savefig`` from this signature, nor
        ``clear_figure``. If False, simply calls ``finalize()``

    markers : string, default: '-d'
        Matplotlib style markers for points on the plot points
        Options: '-,', '-+', '-o', '-*', '-v', '-h', '-d'

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers. These arguments are
        also passed to the ``show()`` method, e.g. can pass a path to save the
        figure to.

    Returns
    -------
    visualizer : ValidationCurve
        The fitted visualizer
    """

    # Initialize the visualizer
    oz = ValidationCurve(
        estimator,
        param_name,
        param_range,
        ax=ax,
        logx=logx,
        groups=groups,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        pre_dispatch=pre_dispatch,
        markers=markers,
    )

    # Fit the visualizer
    oz.fit(X, y)

    # Draw final visualization
    if show:
        oz.show(**kwargs)
    else:
        oz.finalize()

    # Return the visualizer object
    return oz
